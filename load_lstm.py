import torch
from models import LSTMModel

# 加载LSTM模型
input_size = 128 * 128 * 3 # 输入大小
hidden_size = 256 # 隐藏层大小
num_layers = 2 # LSTM层数
output_size = 128 * 128 * 3 # 输出大小

lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size)
lstm_model.load_state_dict(torch.load('/content/drive/My Drive/lstm_model.pth'))
lstm_model.eval()  # 设置模型为评估模式


# 准备输入数据
new_frames = torch.randn(1, input_size)  # 生成一个随机输入序列

# 准备输入数据（假设有一个新的输入序列）
new_frames = torch.randn(1, input_size) # 生成一个随机输入序列

# 使用LSTM模型进行预测
with torch.no_grad():
    prediction = lstm_model(new_frames)
    prediction_np = prediction.squeeze().numpy()

print(f'LSTM model prediction: {prediction_np}')

