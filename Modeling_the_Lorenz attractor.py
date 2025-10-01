import torch
import torch.nn as nn
from torchdiffeq import odeint

class LorenzAttractor(nn.Module):
  def __init__(self, sigma=10, rho=28, beta=8/3):
    super(LorenzAttractor, self).__init__()
    self.sigma = sigma
    self.rho = rho
    self.beta = beta

  def forward(self, t, y):
    x, y, z = y
    dx_dt = self.sigma * (y - x)
    dy_dt = x * (self.rho - z) - y
    dz_dt = x * y - self.beta * z
    return torch.Tensor([dx_dt, dy_dt, dz_dt])

sigma = 10.0
rho = 28.0
beta = 8/3.0
initial_conditions = torch.Tensor([1.0, 1.0, 1.0])

model = LorenzAttractor(sigma, rho, beta)
final_t = 50
t = torch.linspace(0, final_t, 1000)
solution = odeint(model, initial_conditions, t)

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(solution[:, 0], solution[:, 1], solution[:, 2])
plt.show()
