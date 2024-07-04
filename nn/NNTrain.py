from nn.cnn import CNN, CNN_ChangeConv, CNN_ChangeConv2
from nn.dnn import DNN
import torch
from torch.utils.data import DataLoader
from utils import DataLoader as dl
import os.path
from utils import ImageHandler
import matplotlib.pyplot as plt


class NNTrain(object):
    """
    神经网络训练类
    """

    def __init__(self):
        """
        默认构造函数
        """

        self.save_model_path = 'model/NewCNNModel.pkl'  # 模型保存路径
        self.is_use_cuda = True  # 是否使用GPU的cuda
        self.model = None

    def Begin_Train(self):
        """
        开始训练
        """

        data_loader = dl.DataLoader()  # 数据集工具类

        # 可调整参数
        batch_size = 128  # 每次迭代中用于训练模型的样本数。一般为32、64、128、256
        learning_rate = 0.01  # 学习率
        num_epoch = 2  # 训练数据集被完整训练一次的次数
        self.model = CNN.CNN()  # 选择模型
        loss = torch.nn.CrossEntropyLoss()   # 定义损失函数
        # 定义优化器，选择 Adam 优化器来更新模型参数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        train_data = DataLoader(data_loader.get_train_data(), batch_size=batch_size, shuffle=True)
        len_train_data = len(data_loader.get_train_data())  # 训练集数据个数

        if self.is_use_cuda:
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            else:
                print("CUDA核心不可用")

        for i in range(num_epoch):
            print('第{}轮训练开始'.format(i))
            current_batch = 0  # 当前训练批次
            num_correct = 0  # 该轮正确数

            for (x, y) in train_data:
                # 将数据移动到 GPU（如果可用）
                if self.is_use_cuda and torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                outs = self.model(x)
                current_loss = loss(outs, y)

                optimizer.zero_grad()

                current_loss.backward()
                optimizer.step()

                current_batch += 1
                if current_batch % 50 == 0:
                    print('第{}轮，第{}批，损失: {:.4}'.format(i, current_batch, current_loss.data.item()))

                for j, out in enumerate(outs):
                    if torch.argmax(out) == y[j]:
                        num_correct += 1

            current_accuracy = num_correct / len_train_data
            print('第{}轮训练完成，准确率：{:.6f}%'.format(i, current_accuracy * 100))

        self.Save_Model()

    def Begin_Test(self):
        """
        开始测试
        """

        batch_size = 128  # 每次迭代中用于训练模型的样本数
        data_loader = dl.DataLoader()  # 数据集工具类
        test_data = DataLoader(data_loader.get_test_data(), batch_size=batch_size, shuffle=False)
        len_test_data = len(data_loader.get_test_data())

        self.Load_Model()
        self.model.eval()

        num_correct = 0
        with torch.no_grad():
            for (x, y) in test_data:
                if self.is_use_cuda and torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                outs = self.model(x)
                # 计算当前批次正确数
                for j, out in enumerate(outs):
                    if torch.argmax(out) == y[j]:
                        num_correct += 1

        print('正确率：{:.6f}%'.format(100 * num_correct / len_test_data))

    def Save_Model(self):
        """
        保存模型
        """

        print('模型训练完成，正在保存模型')
        torch.save(self.model, self.save_model_path)

    def Load_Model(self):
        """
        加载模型
        """

        if not os.path.exists(self.save_model_path):
            print('模型文件不存在，请重新训练模型')
            return

        self.model = torch.load(self.save_model_path)
        if self.is_use_cuda:
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            else:
                print("CUDA核心不可用")


    def Predict_One_Image(self, image_path):
        """
        预测单张图片

        :param image_path: 图片存放路径
        :return: 预测数字值
        """

        # 创建图像处理类
        image_handler = ImageHandler.ImageHandler()

        self.Load_Model()
        self.model.eval()  # 设置模型为评估模式

        if not os.path.exists(image_path):
            print('图片不存在')
            return

        image = image_handler.convert_image_to_tensor(image_path)

        if self.is_use_cuda and torch.cuda.is_available():
            image = image.cuda()

        with torch.no_grad():  # 关闭梯度计算，提供性能，节省内存
            output = self.model(image)
            predict = torch.argmax(output).item()

            plt.figure(1)
            plt.imshow(image[0].cpu().view(28, 28), cmap='gray')  # 确保将张量移回CPU
            plt.title("Prediction: " + str(predict))
            plt.show()

            return predict
