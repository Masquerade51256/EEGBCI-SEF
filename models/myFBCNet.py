import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import Conv2dWithConstraint, LinearWithConstraint

class FBCNet(nn.Module):
    def __init__(
            self,
            num_channels: int,
            input_time_length: int,  # 明确传入时间长度
            m=32,
            n_band=9,
            temporal_stride=4,
    ):
        super(FBCNet, self).__init__()
        self.temporal_stride = temporal_stride
        self.n_band = n_band
        self.m = m

        # SCB (Spatial Convolution Block)
        self.scb = nn.Sequential(
            Conv2dWithConstraint(n_band, m * n_band, (num_channels, 1), 
                                 groups=n_band, max_norm=2),
            nn.BatchNorm2d(m * n_band),
            Swish()
        )

        # Temporal Layer
        self.temporal_layer = LogVarLayer(dim=-1)  # 在最后一个维度计算方差

        # 动态计算输出特征维度
        with torch.no_grad():
            dummy_input = torch.randn(1, n_band, num_channels, input_time_length)
            dummy_output = self.forward_backbone(dummy_input)
            self.flattened_features = dummy_output.shape[1]  # 获取展平后的特征数
            print(f"[FBCNet] Computed flattened features: {self.flattened_features}")

    def forward_backbone(self, x):
        """
        处理单个样本的前向传播
        
        输入形状: (batch, n_band, num_channels, time_points)
        输出形状: (batch, n_band * m * temporal_stride)
        """
        # x = torch.squeeze(x)  # 移除可能的单维度
        x = self.scb(x)  # 输出: (batch, m*n_band, 1, time_points)
        
        # 动态填充以确保时间维度能被 temporal_stride 整除
        batch_size, channels, height, time_len = x.shape
        
        # 计算需要填充的长度
        remainder = time_len % self.temporal_stride
        if remainder != 0:
            pad_len = self.temporal_stride - remainder
            x = F.pad(x, (0, pad_len))  # 在时间维度末尾填充
            time_len = time_len + pad_len
            # print(f"[FBCNet] Padded time dimension: {time_len - pad_len} -> {time_len}")
        
        # 重塑为时间块
        # 新形状: (batch, channels, temporal_stride, time_len//temporal_stride)
        x = x.reshape([batch_size, channels, self.temporal_stride, 
                      time_len // self.temporal_stride])
        
        x = self.temporal_layer(x)  # 计算每个时间块的方差
        x = torch.flatten(x, start_dim=1)  # 展平
        
        return x

    def forward(self, x):
        return self.forward_backbone(x)


class Classifier(nn.Module):
    def __init__(self, input_features, num_classes):
        super(Classifier, self).__init__()
        self.dense = nn.Sequential(
            LinearWithConstraint(input_features, num_classes, max_norm=0.5),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.dense(x)


class FBCNet_Standard(nn.Module):
    """
    标准化接口的 FBCNet
    输入: (batch, n_band, num_channels, time_points)
    输出: (batch, num_classes)
    """
    def __init__(self,
                 num_classes: int,
                 num_channels: int,
                 input_time_length: int,
                 n_band: int = 9,
                 m: int = 32,
                 temporal_stride: int = 4):
        super(FBCNet_Standard, self).__init__()
        
        print(f"[FBCNet_Standard] Initializing with: "
              f"n_band={n_band}, channels={num_channels}, time_len={input_time_length}")
        
        self.backbone = FBCNet(
            num_channels=num_channels,
            input_time_length=input_time_length,
            m=m,
            n_band=n_band,
            temporal_stride=temporal_stride
        )
        
        self.classifier = Classifier(
            input_features=self.backbone.flattened_features,
            num_classes=num_classes
        )

    def forward(self, x):
        # 可选: 验证输入形状
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (batch, n_band, channels, time), got {x.shape}")
        
        features = self.backbone(x)
        output = self.classifier(features)
        return output


# 激活函数类保持不变
class ActSquare(nn.Module):
    def forward(self, x):
        return torch.square(x)

class ActLog(nn.Module):
    def __init__(self, eps=1e-06):
        super(ActLog, self).__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class LogVarLayer(nn.Module):
    def __init__(self, dim):
        super(LogVarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(x.var(dim=self.dim, keepdim=True), 1e-6, 1e6))