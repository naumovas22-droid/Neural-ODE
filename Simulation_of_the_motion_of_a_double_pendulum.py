import torch
from torch import nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt

class DoublePendulum(nn.Module):
  def __init__(self, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
    super(DoublePendulum, self).__init__()
    self.m1 = m1
    self.m2 = m2
    self.l1 = l1
    self.l2 = l2
    self.g = g

  def forward(self, t, y):
    theta1, theta2, p1, p2 = y
    c = torch.cos(theta1 - theta2)
    s = torch.sin(theta1 - theta2)

    theta1_dot = p1 / (self.m1 * self.l1**2)
    theta2_dot = p2 / (self.m2 * self.l2**2)

    p1_dot = -self.m1 * self.g * self.l1 * torch.sin(theta1) - self.m2 * self.g * self.l2 * torch.sin(theta1)
    p2_dot = -self.m2 * self.g * self.l2 * torch.sin(theta2)

    return torch.stack([theta1_dot, theta2_dot, p1_dot, p2_dot])

model = DoublePendulum()
initial_conditions = torch.tensor([0.5, 0.5, 0.0, 0.0])

t = torch.linspace(0, 15, 1500)

solution = odeint(model, initial_conditions, t)

theta1 = solution[:, 0].detach().numpy()
theta2 = solution[:, 1].detach().numpy()

plt.figure()
plt.plot(t.numpy(), theta1, label='θ_1')
plt.plot(t.numpy(), theta2, label='θ_2')
plt.xlabel('Время')
plt.ylabel('Угол отклонения')
plt.legend()
plt.show()
