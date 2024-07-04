import torch.nn as nn


class CNN_ChangeConv(nn.Module):
    """
    卷积神经网络横向对比实验：更改卷积层参数，两层卷积层+三层全连接
    """

    def __init__(self):
        """
        初始化构造函数
        定义卷积层和全连接层
        """

        super(CNN_ChangeConv, self).__init__()

        self.Conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=25,
                kernel_size=3,
            ),

            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.Conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=25,
                out_channels=50,
                kernel_size=3,
            ),

            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.FC = nn.Sequential(
            nn.Linear(50 * 5 * 5, 1024),
            nn.ReLU(),

            nn.Linear(1024, 128),
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

        x = x.view(x.size(0), -1)
        # 全连接层
        x = self.FC(x)

        return x
