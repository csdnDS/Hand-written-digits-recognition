import torch.nn as nn


class CNN_ChangeConv2(nn.Module):
    """
    卷积神经网络横向对比实验：更改卷积层参数，两层卷积层+一层全连接
    """

    def __init__(self):
        """
        初始化构造函数
        定义卷积层和全连接层
        """

        super(CNN_ChangeConv2, self).__init__()

        self.Conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),

            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.Conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.FC = nn.Sequential(
            nn.Linear(32 * 7 * 7, 10)
        )

    def forward(self, x):
        """
        前向传播
        """

        # 卷积层
        x = self.Conv1(x)
        x = self.Conv2(x)

        x = x.view(x.size(0), -1)
        # 全连接层
        x = self.FC(x)

        return x
