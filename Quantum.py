import time, math
import numpy as np, torch, torch.nn as nn, torch.optim as optim, matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Матрицы Паули образуют базис для 2×2 эрмитовых матриц
sigma_x = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.complex64, device=device)
sigma_y = torch.tensor([[0., -1j], [1j, 0.]], dtype=torch.complex64, device=device)
sigma_z = torch.tensor([[1., 0.], [0., -1.]], dtype=torch.complex64, device=device)
identity = torch.eye(2, dtype=torch.complex64, device=device)

def H_from_coeffs(coeffs):
    c = coeffs
    H = c[...,0].unsqueeze(-1).unsqueeze(-1) * identity
    H = H + c[...,1].unsqueeze(-1).unsqueeze(-1) * sigma_x
    H = H + c[...,2].unsqueeze(-1).unsqueeze(-1) * sigma_y
    H = H + c[...,3].unsqueeze(-1).unsqueeze(-1) * sigma_z
    return H

def schrodinger_rhs(psi, H):
    return -1j * torch.matmul(H, psi.unsqueeze(-1)).squeeze(-1)

def rk4_step_precomputed(psi, dt, H_node, H_mid, H_next):
    # H_node, H_mid, H_next: complex matrices shape (2,2)
    k1 = schrodinger_rhs(psi, H_node)
    k2 = schrodinger_rhs(psi + 0.5*dt*k1, H_mid)
    k3 = schrodinger_rhs(psi + 0.5*dt*k2, H_mid)
    k4 = schrodinger_rhs(psi + dt*k3, H_next)
    return psi + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def integrate_ode_precomputed(psi0, t_grid, coeffs_nodes, coeffs_mid, coeffs_next):
    psi = psi0
    traj = [psi]
    for i in range(len(t_grid)-1):
        dt = (t_grid[i+1] - t_grid[i])
        H_node = H_from_coeffs(coeffs_nodes[i])
        H_mid = H_from_coeffs(coeffs_mid[i])
        H_next = H_from_coeffs(coeffs_next[i])
        psi = rk4_step_precomputed(psi, dt, H_node, H_mid, H_next)
        psi = psi / torch.norm(psi)
        traj.append(psi)
    return torch.stack(traj, dim=0)

def true_H_coeffs_fn(t):
    a0 = 0.1 * torch.sin(0.5 *t)
    ax = 1.0 * torch.cos(1.2 * t + 0.2)
    ay = 0.6 * torch.sin(0.7 * t - 0.3)
    az = 0.8 * torch.cos(0.5 * t + 1.0)
    return torch.stack([a0, ax, ay, az], dim=-1)

t0, t1 = 0.0, 6.0
num_obs = 40
t_grid = torch.linspace(t0, t1, num_obs, device=device)
psi0 = torch.tensor([1.0, 0.0], dtype=torch.complex64, device=device)

with torch.no_grad():
    def H_true_at(t_scalar):
        return H_from_coeffs(true_H_coeffs_fn(t_scalar))
    psi_traj = integrate_ode_precomputed(psi0, t_grid, true_H_coeffs_fn(t_grid), true_H_coeffs_fn((t_grid[:-1]+t_grid[1:])/2), true_H_coeffs_fn(t_grid[1:]))

noise_level = 0.02
noise_real = noise_level * torch.randn_like(psi_traj.real)
noise_imag = noise_level * torch.randn_like(psi_traj.imag)
psi_noisy = psi_traj + (noise_real + 1j*noise_imag)
psi_noisy = psi_noisy / torch.norm(psi_noisy, dim=-1, keepdim=True).clamp(min=1e-8)

class TimeNet(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4)
        )
    def forward(self, t):
        t_in = t.unsqueeze(-1).to(torch.float32)
        return self.net(t_in).to(torch.float32)

class NeuralHamiltonian(nn.Module):
    def __init__(self):
        super().__init__()
        self.tnet = TimeNet(hidden_dim=32)
    def forward(self, t_scalar):
        coeffs = self.tnet(t_scalar)
        return H_from_coeffs(coeffs)

model = NeuralHamiltonian().to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-3)
loss_hist = []

num_epochs = 1000
print_every = 20
start_time = time.time()
for epoch in range(1, num_epochs+1):
    model.train()
    optimizer.zero_grad()
    # Предвычисляем коэффициенты сети в узлах и промежуточных точках (векторно)
    t_nodes = t_grid
    t_mid = (t_grid[:-1] + t_grid[1:]) / 2.0
    t_next = t_grid[1:]
    coeffs_nodes = model.tnet(t_nodes)
    coeffs_mid = model.tnet(t_mid)
    coeffs_next = model.tnet(t_next)
    psi_pred = integrate_ode_precomputed(psi0, t_grid, coeffs_nodes, coeffs_mid, coeffs_next)
    def pauli_expectations(psi_traj):
        exps = []
        for sigma in [sigma_x, sigma_y, sigma_z]:
            vals = torch.sum(torch.conj(psi_traj) * torch.matmul(sigma, psi_traj.unsqueeze(-1)).squeeze(-1), dim=-1)
            exps.append(vals.real)
        return torch.stack(exps, dim=-1)
    exps_true = pauli_expectations(psi_noisy)
    exps_pred = pauli_expectations(psi_pred)
    loss = torch.mean((exps_true - exps_pred)**2)
    loss.backward()
    optimizer.step()
    loss_hist.append(loss.item())
    if epoch % print_every == 0 or epoch == 1:
        print(f"Epoch {epoch}/{num_epochs}, loss={loss.item():.6e}")

end_time = time.time()
print(f"Training finished in {end_time - start_time:.1f} s")

# Оценка
model.eval()
with torch.no_grad():
    coeffs_pred = model.tnet(t_grid)
    coeffs_true = true_H_coeffs_fn(t_grid)
    coeffs_true_centered = coeffs_true.clone()
    coeffs_pred_centered = coeffs_pred.clone()
    coeffs_true_centered[:,0] -= coeffs_true_centered[:,0].mean()
    coeffs_pred_centered[:,0] -= coeffs_pred_centered[:,0].mean()

t_np = t_grid.cpu().numpy()
coeffs_true_np = coeffs_true_centered.cpu().numpy()
coeffs_pred_np = coeffs_pred_centered.cpu().numpy()

plt.figure(figsize=(10,6))
plt.suptitle("Истинные и предсказанные коэффициенты гамильтониана")
labels = ["a0 (I)", "ax (σx)", "ay (σy)", "az (σz)"]
for k in range(3):
    plt.subplot(1,3,k+1)
    plt.plot(t_np, coeffs_true_np[:,k+1], label="Истинная кривая")
    plt.plot(t_np, coeffs_pred_np[:,k+1], label="Предсказание", linestyle='--')
    plt.xlabel("t")
    plt.ylabel(labels[k+1])
    plt.legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.figure(figsize=(6,3))
plt.plot(loss_hist)
plt.xlabel("epoch")
plt.ylabel("Функция потерь")
plt.title("Функция потерь предсказания ⟨σi⟩(t)")
plt.tight_layout()

with torch.no_grad():
    psi_pred = integrate_ode_precomputed(psi0, t_grid, coeffs_pred, coeffs_mid, coeffs_next)
    def pauli_expectations_arr(psi_traj):
        exps = []
        for sigma in [sigma_x, sigma_y, sigma_z]:
            vals = torch.sum(torch.conj(psi_traj) * torch.matmul(sigma, psi_traj.unsqueeze(-1)).squeeze(-1), dim=-1)
            exps.append(vals.real)
        return torch.stack(exps, dim=-1)
    exps_pred = pauli_expectations_arr(psi_pred).cpu().numpy()
    exps_true = pauli_expectations_arr(psi_noisy).cpu().numpy()

plt.figure(figsize=(9,5))
plt.suptitle("⟨σi⟩(t)")
for k, name in enumerate(["⟨σx⟩(t)","⟨σy⟩(t)","⟨σz⟩(t)"]):
    plt.subplot(1,3,k+1)
    plt.plot(t_np, exps_true[:,k], label="Истинная кривая")
    plt.plot(t_np, exps_pred[:,k], linestyle='--', label="Предсказание")
    plt.xlabel("t")
    plt.title(name)
    plt.legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
