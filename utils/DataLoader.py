from torchvision import datasets, transforms

class DataLoader(object):
    """
    数据集加载工具类

    Notes:
        加载训练数据
        加载测试数据
        获取训练、测试数据
    """

    def __init__(self):
        """
        默认构造函数
        """

        self.train_data_path = './data'     # 训练数据集保存的文件夹路径
        self.test_data_path = './data'      # 测试数据集保存的文件夹路径

        # 数据预处理
        self.data_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])
             ])

        # 训练数据集
        self.train_data_set = datasets.MNIST(root=self.train_data_path, transform=self.data_transform, train=True)

        # 测试数据集
        self.test_data_set = datasets.MNIST(root=self.test_data_path, transform=self.data_transform, train=False)

    def get_train_data(self):
        """
        获取训练数据集
        """

        return self.train_data_set

    def get_test_data(self):
        """
        获取测试数据集
        """

        return self.test_data_set
