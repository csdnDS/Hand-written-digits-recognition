import time
from nn import NNTrain

if __name__ == '__main__':

    start = time.time()

    # 创建卷积神经网络训练器
    train = NNTrain.NNTrain()
    train.Begin_Train()

    end = time.time()
    timePass = end - start  # 计算运行时间
    print('Running time: {:.6f} Minute'.format(timePass / 60))
