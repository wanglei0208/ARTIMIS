import torch
import torch.nn as nn
import torch.nn.functional as F

class IdsNet(nn.Module):
    def __init__(self, input_dim=60, num_classes=2, dropout=0.01):
        super(IdsNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class MLP(nn.Module):
    def __init__(self, input_dim=60, hidden_dim1=256, hidden_dim2=128, num_classes=2, dropout=0.3):
        super(MLP, self).__init__()
        # 第一层全连接层
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()  # 激活函数
        self.dropout1 = nn.Dropout(dropout)  # Dropout 防止过拟合
        # 第二层全连接层
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        # 输出层
        self.fc3 = nn.Linear(hidden_dim2, num_classes)
    def forward(self, x):
        # 前向传播
        x = self.fc1(x)  # 输入到第一个隐藏层
        x = self.relu1(x)  # 激活函数
        x = self.dropout1(x)  # Dropout层

        x = self.fc2(x)  # 输入到第二个隐藏层
        x = self.relu2(x)  # 激活函数
        x = self.dropout2(x)  # Dropout层

        x = self.fc3(x)  # 输出层
        return x

class LeNet1D(nn.Module):
    def __init__(self, input_dim=60, num_classes=2):
        super(LeNet1D, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )
        # 👇 计算卷积后特征长度
        self._output_len = self._get_conv_output(input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self._output_len, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)
        )
    def _get_conv_output(self, input_len):
        # 模拟一个 tensor 计算卷积输出长度
        with torch.no_grad():
            x = torch.zeros(1, 1, input_len)  # shape: [batch, channel, length]
            x = self.features(x)
            return x.view(1, -1).size(1)  # flatten 后的长度
    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, input_dim]
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class AlexNet1D(nn.Module):
    def __init__(self,input_dim=60, num_classes=2):
        super(AlexNet1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self._output_len = self._get_conv_output(input_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self._output_len, 4096),  # 调整线性层的输入大小
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    def _get_conv_output(self, input_len):
        # 模拟一个 tensor 计算卷积输出长度
        with torch.no_grad():
            x = torch.zeros(1, 1, input_len)  # shape: [batch, channel, length]
            x = self.features(x)
            return x.view(1, -1).size(1)  # flatten 后的长度
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class AlertNet(nn.Module):
    def __init__(self, input_dim=60, num_classes=2, hidden_dims=[128, 64, 32]):
        super(AlertNet, self).__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())  # 可改为 nn.Tanh() 按需
            in_dim = hidden_dim
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], num_classes)
        )
    def forward(self, x):
        # x: [batch_size, input_dim]
        x = self.features(x)
        x = self.classifier(x)
        return x

class DeepNet(nn.Module):
    def __init__(self, input_dim=60, num_classes=2, dropout_rate=0.01):
        super(DeepNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.net(x)


class AlexNet1D2(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNet1D2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(in_channels=64, out_channels=192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )
        #input [batch_size, 1, 70]
        #conv1 [batch_size, 64, 18]  Pool [batch_size, 64, 8]
        #conv2 [batch_size, 192, 8]  Pool [batch_size, 192, 3]
        #conv3 [batch_size, 384, 3]
        #conv4 [batch_size, 256, 3]
        #conv5 [batch_size, 256, 3]  Pool [batch_size, 256, 1]
        self.avgpool = nn.AdaptiveAvgPool1d(6)  # Adaptive pooling to match feature size [batch_size, 256, 6]

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6, 4096),  # Adjust linear layer input size
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension, becomes [batch_size, 1, 70]
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten feature map
        x = self.classifier(x)
        return x
class VGG11_1D(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG11_1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 4096),  # 根据输入数据调整线性层的输入大小
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度，变为 [batch_size, 1, 70]
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平特征图
        x = self.classifier(x)
        return x


class VGG16_1D(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG16_1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 4096),  # 根据输入数据调整线性层的输入大小
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度，变为 [batch_size, 1, 70]
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平特征图
        x = self.classifier(x)
        return x



class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, self.expansion * out_channels, kernel_size=1,  stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet1D, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Create a list of LSTM layers
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.lstm_layers.append(nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True))
            else:
                self.lstm_layers.append(nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True))
            self.lstm_layers.append(nn.Dropout(0.2))

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        out = x.unsqueeze(1)  # Add a dummy time dimension
        for i in range(self.num_layers):
            out, (h0, c0) = self.lstm_layers[2 * i](out, (h0, c0))
            out = self.lstm_layers[2 * i + 1](out)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        #out = self.sigmoid(out)
        return out