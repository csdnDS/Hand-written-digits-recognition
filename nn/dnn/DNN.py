import torch.nn as nn


class DNN(nn.Module):
    """
    神经网络 DNN 类
    该类继承自 torch.nn.Module，实现其中的接口
    """

    def __init__(self):
        """
        初始化构造函数
        """

        super(DNN, self).__init__()

        self.FC1 = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
        )

        self.FC2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.FC3 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.FC4 = nn.Sequential(
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        """
        前向传播
        """

        x = x.view(x.size(0), -1)
        x = self.FC1(x)
        x = self.FC2(x)
        x = self.FC3(x)
        x = self.FC4(x)

        return x
