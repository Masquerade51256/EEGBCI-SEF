"""审计 v4：纯 sklearn/mne 的 CSP+LDA within-subject 基线，完全绕过项目框架。
对照目标：框架干净协议 CSP+LDA = 51.60%（TWFB 复现实验中的对照组数字）。"""
import numpy as np
import scipy.io as sio
from pathlib import Path
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from mne.decoding import CSP

FS = 500
ROOT = Path("src/datasets/21679035/sourcedata")
MI_WIN = (int(2.0*FS), int(6.0*FS))  # 完整 MI 段 2-6s

def bp(x, lo=8, hi=30):
    b, a = butter(4, [lo/(FS/2), hi/(FS/2)], btype="band")
    return filtfilt(b, a, x, axis=-1)

accs = []
for sub in sorted(ROOT.glob("sub-*")):
    f = sub / f"{sub.name}_task-motor-imagery_eeg.mat"
    if not f.exists():
        continue
    v = sio.loadmat(f)["eeg"]
    eeg = v[0][0][0][:, :30, MI_WIN[0]:MI_WIN[1]]  # 30 EEG 通道（去掉 2 EOG + 事件）
    labels = v[0][0][1].flatten().astype(int) - 1   # 0=left, 1=right
    X = bp(eeg).astype(np.float64)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []
    for tr, te in skf.split(X, labels):
        clf = Pipeline([("csp", CSP(n_components=6, reg=None, log=True, norm_trace=False)),
                        ("lda", LinearDiscriminantAnalysis())])
        clf.fit(X[tr], labels[tr])
        fold_accs.append(clf.score(X[te], labels[te]))
    accs.append(np.mean(fold_accs))
    print(f"{sub.name}: {accs[-1]*100:.1f}%")

accs = np.array(accs)
print(f"\n===== 独立 CSP+LDA 基线（n={len(accs)}）=====")
print(f"mean = {accs.mean()*100:.2f}% ± {accs.std()*100:.2f}%")
print(f"框架对照值: 51.60% ± 13.5%")
