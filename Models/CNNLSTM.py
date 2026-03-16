import torch.nn as nn
import config
import os,yaml
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
    def __init__(self, channels_num, class_num, input_length=2000) -> None:
        super().__init__()
        max_size = 3
        max_stride = 3
        max_padding = 0

        length_after_pooling = math.floor(
            (input_length + 2*max_padding - max_size) / max_stride + 1
        )
        cnn_output_features = 64 * length_after_pooling  # 64是conv3的输出通道数
        # print(f"CNN output features shape: {cnn_output_features}")
        self.cnn = TGCNN(max_size, max_stride, max_padding, channels_num)
        self.lstm = TGLSTM(max_size, max_stride, max_padding, class_num, cnn_output_features)

    def forward(self,x):
        # print(x.shape)
        x = self.cnn(x)
        # x=x.unsqueeze(dim=0)
        # print(x.shape)
        out = self.lstm(x)
        # print(out.shape)
        return out