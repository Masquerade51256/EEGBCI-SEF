# Thwe et al. (2026) 论文复现 — 修改总结

## 日期
2026-04-15

## 复现目标
复现论文：
> Y. Thwe, D. Maneetham, and P. N. Crisnapati, "Integrating EEG Artifact Removal with Deep Learning for Accurate Motor Imagery Classification in Acute Stroke," *Engineering, Technology & Applied Science Research*, vol. 16, no. 1, 2026.

核心复现内容包括：
1. **三种 ICA-based 伪迹去除方案**：EMAR、SASICA、MARA
2. **三种模型结构**：ThweCNN、ThweLSTM、ThweCNNLSTM
3. **忠实还原论文的预处理流程与实验协议**

---

## 一、新增模块与文件

### 1. 伪迹去除处理器 (`preprocessing/artifact_removal/`)

重构了原有的 `artifact_removal.py`，建立了可扩展的 ICA-based 伪迹去除子模块：

- **`base.py`** — `BaseArtifactRemovalProcessor`
  - 提供 MNE `Info` / `Raw` / `Epochs` 转换、ICA fitting/apply 的通用基础设施
  - 内置 **XWStroke 32 通道默认名称与自定义 montage**（根据论文 Table II 极坐标构建）
  - 自动将 `HEOL`/`HEOR` 标记为 `eog` 类型，确保 `mne-icalabel` 的 topoplot 正常提取

- **`emar.py`** — `EMARProcessor`
  - 使用 `mne-icalabel` (ICLabel) 自动分类独立成分
  - 排除 `muscle artifact` / `eye blink` 概率 ≥ 0.9 的成分
  - 匹配论文 MATLAB 代码 `pop_icflag(EEG, [NaN NaN; 0.9 1; 0.9 1; ...])`
  - **关键工程细节**：ICA fitting 在 1–100 Hz 副本上进行，但 artifact 排除应用到**原始未滤波数据**，保留低频信息

- **`sasica.py`** — `SASICAProcessor`
  - 多标准联合判定：EOG 通道 Pearson 相关、时域方差 z-score、时域峰度 z-score、muscle/eye 频段 PSD 比例
  - 在忠实复现配置中采用了**保守阈值**（ muscle 0.5、eye 0.6、z-score 4.0、EOG corr 0.7 ），避免过度去除脑信号

- **`mara.py`** — `MARAProcessor`
  - 提取 IC 空间/时域特征（空间峰度、平均偏差、方差；时域峰度、方差）
  - 使用 `One-Class SVM` 无监督检测异常成分
  - 同时提供 `iclabel_proxy` 策略作为备用

- **`__init__.py`** — 统一导出
- **`artifact_removal_legacy.py`** — 原 `artifact_removal.py` 保留向后兼容

### 2. 模型复现 (`models/ThweCNNLSTM.py`)

严格按论文架构实现三个模型：

- **`ThweCNN`**：`Conv1d(64,k=3)` → ReLU → `MaxPool1d(2)` → Flatten → Dense(50) → ReLU → Linear
- **`ThweLSTM`**：`LSTM(50)` → ReLU → Linear
- **`ThweCNNLSTM`**：`Conv1d(64,k=3)` → ReLU → `MaxPool1d(2)` → reshape → `LSTM(50)` → ReLU → Linear

已注册到模型注册表：`ThweCNN`、`ThweLSTM`、`ThweCNNLSTM`

### 3. 数据集配置

新增/修改了以下数据集 YAML：

- `configs/dataset/XWStroke_singleband_emar.yaml`
- `configs/dataset/XWStroke_singleband_sasica.yaml`
- `configs/dataset/XWStroke_singleband_mara.yaml`
- `configs/dataset/XWStroke_singleband_emar_paper.yaml` — **忠实复现**
- `configs/dataset/XWStroke_singleband_sasica_paper.yaml` — **忠实复现**
- `configs/dataset/XWStroke_singleband_mara_paper.yaml` — **忠实复现**
- `configs/dataset/XWStroke_paper_base.yaml` — 忠实复现的基础模板（无 artifact removal）

### 4. 实验配置 (`configs/reproductions/thwe2026/`)

新增 12 个实验配置：

**原始对比配置（9个，3 artifact × 3 model）：**
- `xwstroke_emar.yaml` / `_cnn.yaml` / `_lstm.yaml`
- `xwstroke_sasica.yaml` / `_cnn.yaml` / `_lstm.yaml`
- `xwstroke_mara.yaml` / `_cnn.yaml` / `_lstm.yaml`

**忠实复现配置（3个）：**
- `xwstroke_emar_paper.yaml`
- `xwstroke_sasica_paper.yaml` — **推荐优先运行**
- `xwstroke_mara_paper.yaml`

以及详细的使用说明文档：`configs/reproductions/thwe2026/README.md`

---

## 二、框架层修改

### 1. `preprocessing/factory.py`
- 注册 `emar`、`sasica`、`mara` 到 `ProcessorFactory`
- 修复导入路径（`filter_banks`、`data_augmentation`）

### 2. `core/base/base_dataset.py`
- 在 resample 后、通道选择前支持 `artifact_removal_pipeline`
- **新增 `EOG_channels` 自动透传**：如果 `dataset.EOG_channels` 存在（如 `[30,31]`），会自动映射为通道名（`["HEOL", "HEOR"]`）并注入到未显式配置 `eog_channels` 的 processor 中

### 3. `core/experiment.py`
- 移除已废弃的 `torch.set_default_tensor_type(torch.cuda.FloatTensor)`，避免与 `mne-icalabel` 产生 CPU/GPU 张量类型冲突

### 4. `trainers/supervised_trainer.py`
- **学习率调度器可配置化**：新增 `self.scheduler_type`，支持 `scheduler.type: none`（恒定学习率），修复此前强制 cosine 退火的问题
- **支持 `k_folds: 1`**：当 `k_folds=1` 时，自动做 stratified 80/20 单次划分，而不是报错
- 修复 `DataLoader` 生成器在 CUDA 环境下的初始化错误

---

## 三、忠实复现配置与原配置的关键差异

| 项目 | 旧配置 | 忠实复现 `_paper.yaml` |
|---|---|---|
| 采样率 | 250 Hz | **500 Hz**（无降采样） |
| 窗口分割 | 2.0s / 0.5s 步长 → 200 样本 | **4.0s / 4.0s 步长 → 40 样本**（匹配论文） |
| 通道数 | 30 EEG | **32 通道**（保留 HEOL/HEOR） |
| 滤波频带 | `[4, 40] Hz` 窄带 | **`[0.5, 100] Hz`** 宽带 |
| 归一化 | z-score | **none** |
| 数据增强 | 噪声/丢弃/扭曲 (p=0.3) | **none** (p=0) |
| SASICA 阈值 | 较激进（排除 ~9 成分） | **保守化**（排除 ~3 成分） |
| weight decay | 0.01 | **0.0** |
| LR scheduler | cosine | **none** |
| batch size | 32 | **16** |
| epochs | 200 | **500** |

---

## 四、主要运行命令

```bash
# 列出已注册模型（确认 ThweCNN/LSTM/CNNLSTM 存在）
python train.py --list-models

# 推荐：最贴近论文的 SASICA + CNN-LSTM 复现
python train.py --config configs/reproductions/thwe2026/xwstroke_sasica_paper.yaml

# 快速 sanity check（1 被试，5 epoch）
python -c "
from core import Config, ExperimentManager
from models import register_all_models
import data, trainers
register_all_models()
config = Config.fromfile('configs/reproductions/thwe2026/xwstroke_sasica_paper.yaml')
config.set('data.subjects', [1])
config.set('training.epochs', 5)
exp = ExperimentManager(config, exp_name='sanity_check')
exp.setup(); exp.run()
"
```

更多批量运行命令请参见 `configs/reproductions/thwe2026/README.md`。

---

# XWStroke 4秒数据支持 - 修改总结

## 问题描述
需要支持读取预处理后的 `_4s.mat` 文件（路径：`src/datasets/21679035/sourcedata/sub-XX/sub-XX_task-motor-imagery_eeg_4s.mat`）

## 修改内容

### 1. 数据集配置文件 (`configs/dataset/XWStroke.yaml`)

添加了以下配置项：

```yaml
dataset:
  # ... 原有配置 ...
  
  # 是否使用4秒预处理数据（_4s.mat文件）
  use_4s_data: true
  
  # 4秒数据的MAT文件中的变量名
  data_var_name: "eeg_data_4s"
  labels_var_name: "labels"
```

### 2. 数据集代码 (`data/datasets.py`)

重写了 `XWStrokeDataset._load_raw_data()` 方法：

- **优先尝试加载 `_4s.mat` 文件**：如果 `use_4s_data=true` 且文件存在，则直接加载预处理后的4秒数据
- **自动回退到原始文件**：如果4秒文件不存在，自动尝试加载原始 `.mat` 文件
- **灵活的数据提取**：根据配置中的变量名从MAT文件中提取数据和标签
- **详细的日志输出**：打印加载的文件路径和数据形状，便于调试

关键代码逻辑：
```python
def _load_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
    # 获取配置
    use_4s_data = self.dataset_info.get('dataset', {}).get('use_4s_data', True)
    data_var_name = self.dataset_info.get('dataset', {}).get('data_var_name', 'eeg_data_4s')
    
    if use_4s_data:
        # 尝试加载4秒数据文件
        subject_file = os.path.join(subject_dir, f'sub-{self.subject_id:02d}_task-motor-imagery_eeg_4s.mat')
        if os.path.exists(subject_file):
            mat_data = sio.loadmat(subject_file)
            data = mat_data[data_var_name]
            labels = mat_data[labels_var_name].flatten()
            return data, labels
    
    # 回退到原始文件
    ...
```

## 测试结果

使用 `xw_adfcnn.yaml` 配置成功运行实验：

```
[Subject 1] Loading 4s preprocessed data: src\datasets\21679035\sourcedata\sub-01\sub-01_task-motor-imagery_eeg_4s.mat
[Subject 1] Loaded 4s data shape: (40, 32, 2000), labels shape: (40,)
Dataset loaded: (200, 1, 30, 500)
Model built: ADFCNN
...
Training Complete
Overall: 0.4950 ± 0.0000
```

## 使用说明

### 运行完整实验

```bash
# 使用所有被试
python train.py --config configs/experiment/xw_adfcnn.yaml

# 使用特定被试
python train.py --config configs/experiment/xw_adfcnn.yaml --name my_exp

# 使用CPU运行
python train.py --config configs/experiment/xw_adfcnn.yaml --device cpu
```

### 切换回原始数据

如果需要使用原始数据（非4秒预处理数据），修改配置：

```yaml
dataset:
  use_4s_data: false  # 设置为false使用原始数据
```

### 预处理新数据

如果还没有 `_4s.mat` 文件，运行预处理脚本：

```bash
python preprocessing_XW.py
```

这会在 `sourcedata/sub-XX/` 目录下生成 `sub-XX_task-motor-imagery_eeg_4s.mat` 文件。

## 文件路径结构

```
src/datasets/21679035/sourcedata/
├── sub-01/
│   ├── sub-01_task-motor-imagery_eeg.mat      # 原始数据
│   └── sub-01_task-motor-imagery_eeg_4s.mat   # 预处理后4秒数据 [使用这个]
├── sub-02/
│   ├── sub-02_task-motor-imagery_eeg.mat
│   └── sub-02_task-motor-imagery_eeg_4s.mat
└── ...
```
