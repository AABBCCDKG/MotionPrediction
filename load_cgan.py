import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from models import Generator, Discriminator

# 加载生成器模型
generator = Generator()
generator.load_state_dict(torch.load('/content/drive/My Drive/generator.pth'))
generator.eval() # 设置模型为评估模式

# 加载判别器模型
discriminator = Discriminator()
discriminator.load_state_dict(torch.load('/content/drive/My Drive/discriminator.pth'))
discriminator.eval() # 设置为评估模式

# 生成随机噪声
noise = torch.randn(1, 100, 1, 1) # 生成随机噪声

# 生成假图像
with torch.no_grad(): # 禁用梯度计算以加速推理过程
    fake_image = generator(noise)

# 将生成的图像转换为NumPy数组并打印范围
fake_image_np = fake_image.squeeze().cpu().numpy()
print(f"Fake image range before normalization: {fake_image_np.min()} to {fake_image_np.max()}")
fake_image_np = (fake_image_np - fake_image_np.min()) / (fake_image_np.max() - fake_image_np.min())
print(f"Fake image range after normalization: {fake_image_np.min()} to {fake_image_np.max()}")

# 将生成的图像转换为可视化形式
img_grid = vutils.make_grid(fake_image, normalize=True)

# 将tensor转换为numpy数组
img_np = img_grid.permute(1, 2, 0).detach().cpu().numpy()

# 显示生成的假图像
plt.figure(figsize=(5, 5))
plt.imshow(img_np)
plt.axis('off')
plt.title("Fake Frame")
plt.show()

# 判别器判别生成的图像
with torch.no_grad():
    prediction = discriminator(fake_image)
    print(f'Discriminator Prediction: {prediction.item()}')