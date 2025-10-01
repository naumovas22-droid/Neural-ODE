import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class NeuralODE(nn.Module):
  def __init__(self):
    super(NeuralODE, self).__init__()
    self.net = nn.Sequential(
      nn.Linear(1, 64),
      nn.Tanh(),
      nn.Linear(64, 1)
    )

  def forward(self, x):
    return self.net(x)

def true_dynamics(x):
  return torch.sin(x)

x_obs = torch.linspace(0, 2 * 3.1416, 100).reshape(-1, 1)
y_obs = true_dynamics(x_obs)

model = NeuralODE()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

e = []
l = []

for epoch in range(1000):
  optimizer.zero_grad()
  y_pred = model(x_obs)
  loss = criterion(y_pred, y_obs)
  loss.backward()
  optimizer.step()
  e.append(epoch)
  l.append(loss.item())

plt.plot(e, l)
plt.xlabel('Количество эпох')
plt.ylabel('Функция потерь')
plt.title('Зависимость функции потерь от количества эпох')
plt.show()

with torch.no_grad():
  y_pred_final = model(x_obs)

plt.figure()
plt.plot(x_obs.detach().numpy(), y_obs.detach().numpy(), label='Истинная динамика')
plt.plot(x_obs.detach().numpy(), y_pred_final.detach().numpy(), label='Предсказанная динамика')
plt.legend()
plt.show()
