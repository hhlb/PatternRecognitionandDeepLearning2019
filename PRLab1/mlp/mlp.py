import torch


# 建立MLP模型
# 该模型为三层 训练效果不佳 准确率保持在0.7以上
class MLP(torch.nn.Module):
  def __init__(self):
    super(MLP, self).__init__()
    self.fc1 = torch.nn.Linear(784, 512)
    self.fc2 = torch.nn.Linear(512, 256)
    self.fc3 = torch.nn.Linear(256, 10)

  # 前向传播 计算梯度
  def forward(self, input):
    input = input.view(-1, 28 * 28)
    output = torch.nn.functional.relu(self.fc1(input))
    output = torch.nn.functional.relu(self.fc2(output))
    return torch.nn.functional.relu(self.fc3(output))
