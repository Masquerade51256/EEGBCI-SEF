import math
import torch
import torch.nn as nn
from models.layers import Conv2dWithConstraint, LazyLinearWithConstraint, PositionalEncodingFourier

class backbone(nn.Module):
    def __init__(self,
                num_channels: int,
                num_bands: int,
                F1=8, D=1, F2= 'auto', P1=4, P2=8, pool_mode= 'mean',
                drop_out=0.25, layer_scale_init_value = 1e-6, nums = 4):
        super(backbone, self).__init__()

        # ... [保留原有的所有层定义，如spectral_1, spectral_2, spatial_1, spatial_2, w_q, w_k, w_v等] ...
        pooling_layer = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]

        pooling_size = 0.3
        hop_size = 0.7

        if F2 == 'auto':
            F2 = F1 * D

        # Spectral
        self.spectral_1 = nn.Sequential(
            Conv2dWithConstraint(num_bands, F1, kernel_size=[1, 125], padding='same',  max_norm=2.),
            nn.BatchNorm2d(F1),
            )
        self.spectral_2 = nn.Sequential(
            Conv2dWithConstraint(num_bands, F1, kernel_size=[1, 30], padding='same', max_norm=2.),
            nn.BatchNorm2d(F1),
            )

        # Spatial
        self.spatial_1 = nn.Sequential(
            Conv2dWithConstraint(F1, F2, (num_channels, 1), padding=0, groups=F1, bias=False, max_norm=2.), # 注意: 输入通道应为F1
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.Dropout(drop_out),
            Conv2dWithConstraint(F2, F2, kernel_size=[1, 1], padding='valid',
                                 max_norm=2.),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            pooling_layer((1, 32), stride=32),
            nn.Dropout(drop_out),
        )

        self.spatial_2 = nn.Sequential(
            Conv2dWithConstraint(F1, F2, kernel_size=[num_channels, 1], padding='valid',
                                                 max_norm=2.), # 注意: 输入通道应为F1
            nn.BatchNorm2d(F2),
            ActSquare(),
            pooling_layer((1, 75), stride=25),
            ActLog(),
            nn.Dropout(drop_out),

        )

        self.flatten = nn.Flatten()
        self.drop = nn.Dropout(drop_out)
        # 注意力层的线性变换，其输入特征数需要是 F2 * 某个值，但这里我们先按原结构，后续在forward中调整
        # 注意：由于spatial_1和spatial_2的输出在时间维拼接，其第二维（通道维）都是F2，但时间维长度可能不同。
        # 我们需要在forward中计算拼接后展平的特征数，这里先占位。
        self._attention_feat_dim = None
        self.w_q = nn.Linear(0, 0) # 占位，将在首次前向传播后初始化
        self.w_k = nn.Linear(0, 0)
        self.w_v = nn.Linear(0, 0)

    def forward(self, x):
        x_1 = self.spectral_1(x)
        x_2 = self.spectral_2(x)

        x_filter_1 = self.spatial_1(x_1) # 形状: [B, F2, 1, T1]
        x_filter_2 = self.spatial_2(x_2) # 形状: [B, F2, 1, T2]
        
        # 在最后一个维度（时间维）上拼接
        x_concat = torch.cat((x_filter_1, x_filter_2), dim=3) # 形状: [B, F2, 1, T1+T2]
        
        # 为注意力机制准备: [B, N, C]，其中N=1*(T1+T2), C=F2
        B, C, H, W = x_concat.shape
        x_for_attn = x_concat.reshape(B, C, H * W).permute(0, 2, 1)  # -> [B, H*W, C] 即 [B, N, C]
        
        # 动态初始化注意力层的线性变换（如果尚未初始化）
        if self._attention_feat_dim is None:
            self._attention_feat_dim = x_for_attn.size(-1) # C
            self.w_q = nn.Linear(self._attention_feat_dim, self._attention_feat_dim).to(x.device)
            self.w_k = nn.Linear(self._attention_feat_dim, self._attention_feat_dim).to(x.device)
            self.w_v = nn.Linear(self._attention_feat_dim, self._attention_feat_dim).to(x.device)
        
        B, N, C = x_for_attn.shape
        
        q = self.w_q(x_for_attn) # [B, N, C]
        k = self.w_k(x_for_attn) # [B, N, C]
        v = self.w_v(x_for_attn) # [B, N, C]
        
        # 简化的自注意力（无多头，无缩放？原代码有缩放）
        d_k = q.size(-1)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(d_k) # [B, N, N]
        attn = attn.softmax(dim=-1)
        
        attended = attn @ v # [B, N, C]
        
        # 残差连接
        x_after_attn = x_for_attn + self.drop(attended) # [B, N, C]
        
        # 恢复为 [B, C, H, W] 形状以进行可能的后续处理，但这里我们直接展平
        # 注意: H=1, W=N
        x_flat = x_after_attn.permute(0, 2, 1).reshape(B, C, 1, N) # 回到 [B, C, 1, N]
        x_flat = self.flatten(x_flat) # 展平 -> [B, C * 1 * N] 即 [B, C * N]
        
        return x_flat # 返回展平的特征向量


class classifier(nn.Module):
    def __init__(self, input_features: int, num_classes: int):
        super(classifier, self).__init__()
        # 使用一个简单的线性层（全连接层）进行分类
        self.dense = nn.Sequential(
            nn.Linear(input_features, num_classes),
            # 注意：通常与CrossEntropyLoss搭配时，不在这里添加LogSoftmax
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # x 的形状应为 [batch, input_features]
        x = self.dense(x)
        # 输出形状: [batch, num_classes]
        return x


class ADFCNN(nn.Module):   
    def __init__(self,
                num_classes: int,
                num_channels: int,
                num_bands: int,
                input_length: int):
        super(ADFCNN, self).__init__()
        self.backbone = backbone(num_channels=num_channels, num_bands=num_bands)

        # 关键：动态获取backbone输出的特征维度
        # 创建一个虚拟输入来探测特征维度
        with torch.no_grad():
            # 假设输入形状为 [1, num_bands, num_channels, timepoints]，这里timepoints需与训练数据一致（例如2000）
            dummy_input = torch.randn(1,num_bands, num_channels, input_length)
            dummy_features = self.backbone(dummy_input)
            flattened_features = dummy_features.shape[1]

        self.classifier = classifier(input_features=flattened_features, num_classes=num_classes)

    def forward(self, x):
        # 前向传播
        features = self.backbone(x)  # features形状: [batch, flattened_features]
        out = self.classifier(features)  # out形状: [batch, num_classes]
        return out


# ... [ActSquare 和 ActLog 类保持不变] ...
class ActSquare(nn.Module):
    def __init__(self):
        super(ActSquare, self).__init__()
        pass
    def forward(self, x):
        return torch.square(x)

class ActLog(nn.Module):
    def __init__(self, eps=1e-06):
        super(ActLog, self).__init__()
        self.eps = eps
    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))