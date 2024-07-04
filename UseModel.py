from nn import NNTrain

# 使用模型
predict = NNTrain.NNTrain()
predict_num = predict.Predict_One_Image('image/1.png')
print('预测的数字为：{}'.format(predict_num))
