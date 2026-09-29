"""批量扫描并修复 SHU EDF 头中缺失的 physical min/max 字段。
对读取失败的文件，以同被试任一可读文件为供体，仅填补空白字段。
原文件备份为 .broken。幂等：已修复/可读的文件直接跳过。
"""
from pathlib import Path
import mne

mne.set_log_level("ERROR")
EDF_DIR = Path("src/datasets/SHU/edf_raw/edf")

LAYOUT = [("label", 16), ("transducer", 80), ("phys_dim", 8),
          ("phys_min", 8), ("phys_max", 8), ("dig_min", 8),
          ("dig_max", 8), ("prefilter", 80), ("n_samples", 8), ("reserved", 32)]

def parse_header(path):
    with open(path, "rb") as f:
        fixed = f.read(256)
        ns = int(fixed[252:256].decode().strip())
        hdr_bytes = int(fixed[184:192].decode().strip())
        f.seek(256)
        fields = {}
        for name, w in LAYOUT:
            fields[name] = [f.read(w) for _ in range(ns)]
    return fixed, ns, hdr_bytes, fields

def readable(path):
    try:
        mne.io.read_raw_edf(str(path), preload=False, verbose=False)
        return True
    except Exception:
        return False

def repair(bad, donor):
    fixed_b, ns_b, hdr_b, f_b = parse_header(bad)
    _, ns_d, _, f_d = parse_header(donor)
    assert ns_b == ns_d, (bad, donor)
    new_fields = {k: list(v) for k, v in f_b.items()}
    patched = 0
    for ch in range(ns_b):
        for key in ("phys_min", "phys_max"):
            if not f_b[key][ch].decode("ascii", "ignore").strip():
                new_fields[key][ch] = f_d[key][ch]
                patched += 1
    if patched == 0:
        print(f"!! {bad.name} 读取失败但无空字段，跳过（需人工检查）")
        return False
    header = bytearray(fixed_b)
    for name, w in LAYOUT:
        for ch in range(ns_b):
            header.extend(new_fields[name][ch])
    with open(bad, "rb") as f:
        f.seek(hdr_b)
        body = f.read()
    backup = bad.with_suffix(".edf.broken")
    bad.rename(backup)
    with open(bad, "wb") as f:
        f.write(bytes(header))
        f.write(body)
    print(f"修复 {bad.name}: {patched} 字段（供体 {donor.name}）")
    return True

files = sorted(EDF_DIR.glob("*.edf"))
print(f"共 {len(files)} 个 EDF")
by_subject = {}
for f in files:
    by_subject.setdefault(f.name.split("_")[0], []).append(f)

n_ok, n_fixed, n_failed = 0, 0, 0
for sub, sub_files in sorted(by_subject.items()):
    good = next((f for f in sub_files if readable(f)), None)
    for f in sub_files:
        if readable(f):
            n_ok += 1
            continue
        if good is None:
            print(f"!! {sub} 全部不可读，跳过")
            n_failed += 1
            continue
        if repair(f, good):
            n_fixed += 1 if readable(f) else 0
            n_failed += 0 if readable(f) else 1
print(f"\n完成：原可读 {n_ok}，新修复 {n_fixed}，仍失败 {n_failed}")
