import torch
import noise
from models import Generator, Discriminator
import numpy as np
import matplotlib.pyplot as plt

# 加载生成器模型
generator = Generator()
generator.load_state_dict(torch.load('/content/drive/My Drive/generator.pth'))
generator.eval()  # 设置模型为评估模式

# 加载判别器模型
discriminator = Discriminator()
discriminator.load_state_dict(torch.load('/content/drive/My Drive/discriminator.pth'))
discriminator.eval() # 设置为评估模式

# 生成随机噪声
with torch.no_grad(): # 禁用梯度计算以加速推理过程
    fake_image = generator(noise)

# 将生成的图像转换为NumPy数组
fake_image_np = fake_image.squeeze().numpy()

# 显示生成的图像
plt.imshow(np.transpose(fake_image_np, (1, 2, 0)))
plt.show()

# 判别器判别生成的图像
with torch.no_grad():
    prediction = discriminator(fake_image)
    print(f'Discriminator Prediction: {prediction.item()}')