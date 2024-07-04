import torch.nn as nn


class CNN(nn.Module):
    """
    卷积神经网络 CNN 类
    该类继承自 torch.nn.Module，实现其中的接口
    """

    def __init__(self):
        """
        初始化构造函数
        定义卷积层和全连接层
        """

        super(CNN, self).__init__()

        self.Conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.Conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.Conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.FC = nn.Sequential(
            nn.Linear(128 * 3 * 3, 768),
            nn.ReLU(),

            nn.Linear(768, 128),
            nn.ReLU(),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        """
        前向传播
        """

        # 卷积层
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)

        x = x.view(x.size(0), -1)
        # 全连接层
        x = self.FC(x)

        return x
