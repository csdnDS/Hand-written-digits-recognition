from torchvision import transforms
from PIL import Image

class ImageHandler(object):
    """
    图像处理类
    """

    def convert_image_to_tensor(self, image_path):
        """
        将图片转化为 张量（Tensor）格式

        :param image_path: 图片存放路径
        :return: 张量（Tensor）格式的数据
        """

        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        image = Image.open(image_path)
        image = self.Reverse_Image(image)
        image = transform(image).unsqueeze(0)
        return image


    def Reverse_Image(self, image):
        """
        反转图像颜色
        对于MNIST数据集，数字为白色，底色为黑色
        为了使得预测效果准确，在将数字图片转化为灰度图像后，还需要保证底色为黑色，数字为白色
        此函数识别图片的亮度，如果亮度超出阈值则自动转化图片底色

        :param image: 待转化的图片
        :return: 转化后的图片
        """

        grayscale = image.convert('L')
        pixels = list(grayscale.getdata())
        avg_brightness = sum(pixels) / len(pixels)

        # 如果平均亮度大于128阈值，则认为背景是亮的，进行反转
        if avg_brightness > 128:
            image = Image.eval(image, lambda x: 255 - x)
        return image
