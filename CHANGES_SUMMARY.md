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
