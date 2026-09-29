"""修复 SHU sub-005_ses-04 EDF 头中缺失的 physical min/max 字段。
策略：从同被试 ses-03（同设备同设置）读取每通道 physical min/max 参考值，
仅填补坏文件中为空的字段，其余字节不动。原文件备份为 .broken。
EDF 头布局（BDF/EDF 规范）：256B 通用头 + 每字段每通道定长 ASCII。
"""
from pathlib import Path

BAD = Path("src/datasets/SHU/edf_raw/edf/sub-005_ses-04_task_motorimagery_eeg.edf")
GOOD = Path("src/datasets/SHU/edf_raw/edf/sub-005_ses-03_task_motorimagery_eeg.edf")

def parse_header(path):
    with open(path, "rb") as f:
        fixed = f.read(256)
        n_channels = int(fixed[252:256].decode().strip())
        header_bytes = int(fixed[184:192].decode().strip())
        f.seek(256)
        fields = {}
        layout = [("label", 16), ("transducer", 80), ("phys_dim", 8),
                  ("phys_min", 8), ("phys_max", 8), ("dig_min", 8),
                  ("dig_max", 8), ("prefilter", 80), ("n_samples", 8), ("reserved", 32)]
        for name, width in layout:
            fields[name] = [f.read(width) for _ in range(n_channels)]
    return fixed, n_channels, header_bytes, fields

_, ns_good, _, good_fields = parse_header(GOOD)
fixed_bad, ns_bad, hdr_bad, bad_fields = parse_header(BAD)
assert ns_good == ns_bad == 32, (ns_good, ns_bad)

patched = 0
new_fields = {k: list(v) for k, v in bad_fields.items()}
for ch in range(ns_bad):
    for key in ("phys_min", "phys_max"):
        if not bad_fields[key][ch].decode("ascii", "ignore").strip():
            new_fields[key][ch] = good_fields[key][ch]
            patched += 1
            print(f"ch{ch} {key}: '' -> {good_fields[key][ch].decode()}")

if patched == 0:
    print("未发现空字段？请人工检查")
    raise SystemExit(1)

# 重组头部：256B 固定 + 按布局写回各字段
layout = [("label", 16), ("transducer", 80), ("phys_dim", 8),
          ("phys_min", 8), ("phys_max", 8), ("dig_min", 8),
          ("dig_max", 8), ("prefilter", 80), ("n_samples", 8), ("reserved", 32)]
header = bytearray(fixed_bad)
for name, width in layout:
    for ch in range(ns_bad):
        val = new_fields[name][ch]
        assert len(val) == width, (name, ch, len(val))
        header.extend(val)

with open(BAD, "rb") as f:
    f.seek(hdr_bad)
    body = f.read()

backup = BAD.with_suffix(".edf.broken")
BAD.rename(backup)
with open(BAD, "wb") as f:
    f.write(bytes(header))
    f.write(body)
print(f"patched {patched} fields; backup -> {backup}")

# 验证
import mne
mne.set_log_level("ERROR")
raw = mne.io.read_raw_edf(str(BAD), preload=True, verbose=False)
d = raw.get_data()
print("verify:", raw.info["sfreq"], "Hz,", len(raw.ch_names), "ch,",
      "range %.1f to %.1f" % (d.min() * 1e6, d.max() * 1e6), "uV")
