import torch
import torch.nn as nn
import math


class TGCNN(nn.Module):
    def __init__(self,max_size,max_stride,max_padding,channels_num) -> None:
        super().__init__()

        self.conv1_frez_cal = nn.Sequential(
            nn.Conv1d(channels_num, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace = True),
        )
        self.conv2_frez_ol = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace = True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool1d(kernel_size=max_size, stride=max_stride, padding=max_padding),
        )
        self.flat = nn.Flatten()

    def forward(self,x):
        # print(x.shape)
        x = self.conv1_frez_cal(x)
        x = self.conv2_frez_ol(x)
        x = self.conv3(x)
        out = self.flat(x)
        return out

class TGLSTM(nn.Module):
    def __init__(self,max_size,max_stride,max_padding,nclass,feat_num=5335) -> None:
        super().__init__()
        self.fc= nn.Linear(feat_num, 64)

        self.bn = nn.BatchNorm1d(64)
        self.lstm = nn.LSTM(input_size=64,hidden_size=32,num_layers=1,batch_first= True,bidirectional=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.classify = nn.Linear(32,nclass)
        self.softmax = nn.Softmax(1)
        
    def forward(self,x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.bn(x)
        x=x.unsqueeze(dim=1)
        # print(x.shape)
        x,(h_s,h_c) = self.lstm(x,None)
        # print(x.shape)
        x = self.relu(x[:,-1])
        x = self.dropout(x)
        x = self.classify(x)
        out = self.softmax(x)
        return out


class CNNLSTM(nn.Module):
    def __init__(self, 
                 num_channels: int, 
                 num_classes: int, 
                 num_bands: int,
                 input_length: int) -> None:
        super().__init__()
        max_size = 3
        max_stride = 3
        max_padding = 0

        length_after_pooling = math.floor(
            (input_length + 2*max_padding - max_size) / max_stride + 1
        )
        cnn_output_features = 64 * length_after_pooling  # 64是conv3的输出通道数
        # print(f"CNN output features shape: {cnn_output_features}")

        self.band_cnns = nn.ModuleList([
            TGCNN(max_size, max_stride, max_padding, num_channels) 
            for _ in range(num_bands)
        ])
        
        
        total_cnn_features = cnn_output_features * num_bands
        self.fusion_fc = nn.Linear(total_cnn_features, 256) # 融合后维度可调
        self.relu = nn.ReLU()
        
        self.lstm = TGLSTM(max_size, max_stride, max_padding, num_classes, 256)

    def forward(self,x):
        # print(x.shape)
        batch_size, C, B, T = x.shape
        band_features = []
        for b in range(B):
            band_data = x[:, :, b, :]
            band_features.append(self.band_cnns[b](band_data))
        x = torch.cat(band_features, dim=1)
        x = self.fusion_fc(x)
        x = self.relu(x)
        out = self.lstm(x)
        # print(out.shape)
        return out
    

    import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiBand_CNNLSTM(nn.Module):
    """
    专为多频带EEG数据 (batch, channels, bands, time) 设计的新架构。
    设计思路：
    1. 用2D卷积处理“通道×频带”的二维特征图
    2. 用1D卷积提取时间维度特征
    3. 用LSTM捕获长时依赖
    4. 加入残差连接和注意力机制提升性能
    """
    
    def __init__(self, 
                 channels_num: int,       # 通道数 (C)
                 bands_num: int,          # 频带数 (B)
                 class_num: int,          # 分类数
                 input_length: int,       # 时间点数 (T)
                 dropout_rate: float = 0.3,
                 use_attention: bool = True) -> None:
        super().__init__()
        
        self.channels_num = channels_num
        self.bands_num = bands_num
        self.use_attention = use_attention
        
        # ========== 第一阶段：频带-通道特征提取 (2D卷积) ==========
        # 将 (C, B) 视为2D特征图，用2D卷积提取空间-频带联合特征
        self.conv2d_block = nn.Sequential(
            # 输入: (batch, 1, C, B) - 添加通道维度给2D卷积
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
            nn.Dropout2d(dropout_rate/2),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),  # 仅在频带维度下采样
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
        )
        
        # 计算经过2D卷积后的特征维度
        # 频带维度: B -> B/2 (由于MaxPool2d(kernel_size=(1,2)))
        # 通道维度: C保持不变 (因为padding=same)
        bands_after_pool = bands_num // 2
        self.conv2d_out_channels = 64
        conv2d_feat_dim = self.conv2d_out_channels * channels_num * bands_after_pool
        
        # ========== 第二阶段：时间特征提取 (1D卷积) ==========
        # 将2D特征展平为时间序列，用1D卷积处理
        self.conv1d_block = nn.Sequential(
            # 输入形状将在forward中重塑为: (batch, conv2d_out_channels*C, bands_after_pool, time)
            # 但我们将其视为多个特征通道的时间序列
        )
        
        # 更精确的1D卷积设计：处理重塑后的特征
        # 我们将conv2d的输出重塑为 (batch, new_channels, time) 格式
        # 其中 new_channels = conv2d_out_channels * channels_num * bands_after_pool / input_length
        # 但更稳定的做法是使用自适应池化统一时间维度
        
        self.time_conv = nn.Sequential(
            nn.Conv1d(in_channels=conv2d_feat_dim // input_length, 
                     out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ELU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ELU(inplace=True),
        )
        
        # 计算时间卷积后的特征维度
        time_after_pool = input_length // 2
        conv1d_out_features = 256 * time_after_pool
        
        # ========== 第三阶段：频带注意力机制 (可选) ==========
        if use_attention:
            self.band_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
                nn.Flatten(),
                nn.Linear(self.conv2d_out_channels, self.conv2d_out_channels // 4),
                nn.ReLU(inplace=True),
                nn.Linear(self.conv2d_out_channels // 4, self.conv2d_out_channels),
                nn.Sigmoid()
            )
        
        # ========== 第四阶段：LSTM时序建模 ==========
        # 将卷积特征转换为序列输入LSTM
        self.lstm_input_size = 256
        self.lstm_hidden_size = 128
        self.num_lstm_layers = 1
        
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.num_lstm_layers,
            batch_first=True,
            bidirectional=True,  # 使用双向LSTM
            dropout=dropout_rate if self.num_lstm_layers > 1 else 0
        )
        
        # ========== 第五阶段：分类头 ==========
        lstm_output_features = self.lstm_hidden_size * 2  # 双向LSTM
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_features, 64),
            nn.BatchNorm1d(64),
            nn.ELU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, class_num)
        )
        
        # 自适应池化层，统一时间维度
        self.adaptive_pool = nn.AdaptiveAvgPool2d((None, 50))  # 将时间维度统一为50
        
        # 残差连接
        self.residual_conv = nn.Conv2d(1, 64, kernel_size=1) if channels_num == 64 else None
        
    def forward(self, x):
        """
        前向传播
        输入 x 形状: (batch, C, B, T)
        C: 通道数, B: 频带数, T: 时间点数
        """
        batch_size, C, B, T = x.shape
        
        # ===== 1. 初始处理 =====
        # 添加通道维度: (batch, 1, C, B, T)
        x = x.unsqueeze(1)
        
        # ===== 2. 处理每个时间点 =====
        # 我们将时间维度移到末尾，方便逐时间点处理
        x = x.permute(0, 1, 4, 2, 3)  # (batch, 1, T, C, B)
        
        # 用于存储每个时间点的特征
        time_features = []
        
        # 对每个时间点应用2D卷积
        for t in range(T):
            # 获取当前时间片的频带-通道特征图: (batch, 1, C, B)
            time_slice = x[:, :, t, :, :]
            
            # 2D卷积提取频带-通道特征
            conv2d_out = self.conv2d_block(time_slice)  # (batch, 64, C, B/2)
            
            # 应用频带注意力 (如果启用)
            if self.use_attention:
                attention_weights = self.band_attention(conv2d_out)  # (batch, 64)
                attention_weights = attention_weights.view(batch_size, 64, 1, 1)
                conv2d_out = conv2d_out * attention_weights
            
            # 展平空间-频带维度: (batch, 64*C*(B/2))
            flat_feat = conv2d_out.reshape(batch_size, -1)
            time_features.append(flat_feat.unsqueeze(1))  # 添加时间维度
        
        # 组合所有时间点特征: (batch, T, 64*C*(B/2))
        temporal_seq = torch.cat(time_features, dim=1)
        
        # ===== 3. 时间维度处理 =====
        # 将特征序列转换为适合LSTM的形状
        # 首先通过全连接层降维
        seq_length = temporal_seq.size(1)
        feat_dim = temporal_seq.size(2)
        
        # 重塑为适合LSTM的序列: (batch, T, lstm_input_size)
        lstm_input = temporal_seq.view(batch_size, seq_length, -1)
        
        # 如果需要，通过线性层调整维度
        if feat_dim != self.lstm_input_size:
            lstm_input = lstm_input.reshape(batch_size * seq_length, -1)
            adjust_layer = nn.Linear(feat_dim, self.lstm_input_size).to(lstm_input.device)
            lstm_input = adjust_layer(lstm_input)
            lstm_input = lstm_input.reshape(batch_size, seq_length, self.lstm_input_size)
        
        # ===== 4. LSTM时序建模 =====
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)
        
        # 取最后一个时间步的输出 (双向LSTM的拼接)
        lstm_last = lstm_out[:, -1, :]  # (batch, lstm_hidden_size*2)
        
        # ===== 5. 分类 =====
        output = self.classifier(lstm_last)
        
        return output


class Simplified_MultiBand_CNNLSTM(nn.Module):
    """
    简化版本：更直接的处理方式，适合快速实验
    核心思想：将频带视为额外的通道维度
    """
    
    def __init__(self, 
                 channels_num: int,
                 bands_num: int,
                 class_num: int,
                 input_length: int,
                 dropout_rate: float = 0.3) -> None:
        super().__init__()
        
        # 将频带维度合并到通道维度
        total_channels = channels_num * bands_num
        
        # 1D卷积处理时间序列
        self.conv1d_layers = nn.Sequential(
            # 第一层卷积
            nn.Conv1d(in_channels=total_channels, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ELU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # 第二层卷积
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ELU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # 第三层卷积
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
        )
        
        # 计算卷积后的时间维度
        conv_output_length = input_length // 4  # 经过两次2倍下采样
        conv_output_features = 512 * conv_output_length
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),  # 双向LSTM: 256 * 2=512
            nn.BatchNorm1d(128),
            nn.ELU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, class_num)
        )
        
    def forward(self, x):
        """
        输入: (batch, C, B, T)
        输出: (batch, class_num)
        """
        batch_size, C, B, T = x.shape
        
        # 1. 合并通道和频带维度
        x = x.reshape(batch_size, C * B, T)  # (batch, C*B, T)
        
        # 2. 1D卷积提取特征
        conv_out = self.conv1d_layers(x)  # (batch, 512, T/4)
        
        # 3. 准备LSTM输入: 将特征维度转为序列
        lstm_input = conv_out.permute(0, 2, 1)  # (batch, T/4, 512)
        
        # 4. LSTM处理
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)
        
        # 5. 取最后时间步的双向特征
        lstm_last = lstm_out[:, -1, :]  # (batch, 512)
        
        # 6. 分类
        output = self.classifier(lstm_last)
        
        return output