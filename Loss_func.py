import torch
from torch import nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt

class NeuralODE(nn.Module):
    def __init__(self):
        super(NeuralODE, self).__init__()
        self.func = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2)
        )
    def forward(self, t, x):
        return self.func(x)

neural_ode = NeuralODE()
optimizer = torch.optim.Adam(neural_ode.parameters(), lr=0.01)

def loss_func(y_pred, y_true):
    return torch.mean((y_pred - y_true)**2)

t = torch.linspace(0, 1, 100)
x = torch.randn(100, 2)
y_true = torch.randn(100, 2)
e = []
l = []
for epoch in range(1000):
    optimizer.zero_grad()
    y_pred = odeint(neural_ode, x[0], t)
    loss = loss_func(y_pred, y_true)
    loss.backward()
    optimizer.step()
    e.append(epoch)
    l.append(loss.item())
plt.plot(e, l)
plt.xlabel('Количество эпох')
plt.ylabel('Функция потерь')
plt.title('Зависимость функции потерь от количества эпох')
plt.show()
