"""
SMOKE TEST — Publication-Grade PINN Architecture & Noise Modeling
=============================================================================
Tests the upgraded PINN on Aluminum, Copper, and Brass at the highest noise
tier (sigma = 0.5623) across 10 trials.

Key Improvements:
1. Softplus parameterization mu = softplus(z) + 1e-4 (no gradient vanishing/explosion).
2. Physical log-normal multiplicative noise (zero unphysical clipping artifacts).
3. Robust pairwise-median initial slope estimation (Theil-Sen principle).
4. Stationary metric for early stopping (unweighted data + physics loss).
5. Pre-step checkpointing to protect model weights from overshoots.
6. Unique material-dependent PRNG seeding.
"""

import os
import copy
import warnings
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.stats import linregress, qmc

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ALT_DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "PINN-for-gamma-rays", "data")

MATERIALS_TO_TEST = ["aluminum", "copper", "brass"]
NOISE_STD = 0.5623
N_SMOKE_TRIALS = 10
PINN_EPOCHS = 3000
M_COLLOC = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = torch.float32

print(f"Publication-Grade Smoke Test running on compute device: {device}")
print("=" * 75)


def resolve_file(mat_key):
    candidates = [
        os.path.join(DATA_DIR, f"{mat_key}.csv"),
        os.path.join(ALT_DATA_DIR, f"{mat_key}.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def robust_initial_slope(x_sub, y_noisy):
    """
    Pairwise median slope in log-space (Theil-Sen estimator).
    Robust against high-noise fluctuations.
    """
    log_y = np.log(np.maximum(y_noisy, 1e-12))
    n = len(x_sub)
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = x_sub[j] - x_sub[i]
            if abs(dx) > 1e-6:
                slopes.append(-(log_y[j] - log_y[i]) / dx)
    if slopes:
        med_slope = float(np.median(slopes))
        if 1e-4 <= med_slope <= 2.0:
            return med_slope

    try:
        slope, _, _, _, _ = linregress(x_sub, log_y)
        if -slope > 1e-4:
            return float(-slope)
    except Exception:
        pass

    return 0.05


# ==========================================
# UPGRADED PINN ARCHITECTURE
# ==========================================
class PINN(nn.Module):
    def __init__(self, initial_mu, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        self.mu_floor = 1e-4
        target_mu = max(float(initial_mu) - self.mu_floor, 1e-4)
        if target_mu < 20.0:
            z_init = np.log(np.exp(target_mu) - 1.0)
        else:
            z_init = target_mu
        self.z_mu = nn.Parameter(torch.tensor(z_init, dtype=TORCH_DTYPE))

    def get_mu(self):
        return torch.nn.functional.softplus(self.z_mu) + self.mu_floor

    def forward(self, x):
        return self.net(x)


def train_pinn_lhs(x_train, y_train, x_max, initial_mu, trial, epochs=PINN_EPOCHS):
    x_mean, x_std = x_train.mean(), x_train.std()
    y_log = np.log(y_train)
    y_mean, y_std = y_log.mean(), y_log.std()

    x_data_t = torch.tensor((x_train - x_mean) / x_std, dtype=TORCH_DTYPE).view(-1, 1).to(device)
    y_data_t = torch.tensor((y_log - y_mean) / y_std, dtype=TORCH_DTYPE).view(-1, 1).to(device)

    sampler = qmc.LatinHypercube(d=1, seed=42 + trial)
    lhs_samples = sampler.random(n=M_COLLOC) * x_max
    x_colloc_t = torch.tensor((lhs_samples - x_mean) / x_std, dtype=TORCH_DTYPE).view(-1, 1).to(device)
    x_colloc_t.requires_grad_(True)

    model = PINN(initial_mu=initial_mu).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_loss = float('inf')
    best_model_state = None
    lambda_phys = 1.0
    alpha_ema = 0.9

    for epoch in range(epochs):
        # Adaptive gradient balancing every 10 epochs
        if epoch % 10 == 0:
            optimizer.zero_grad()
            y_pred_data_tmp = model(x_data_t)
            data_loss_tmp = torch.mean((y_pred_data_tmp - y_data_t) ** 2)
            data_loss_tmp.backward(retain_graph=True)
            grads_data = [p.grad.abs().max() for p in model.net.parameters() if p.grad is not None]
            max_grad_data = torch.max(torch.stack(grads_data)) if grads_data else torch.tensor(1.0).to(device)

            optimizer.zero_grad()
            y_pred_colloc_tmp = model(x_colloc_t)
            dy_dx_tmp = torch.autograd.grad(y_pred_colloc_tmp, x_colloc_t, torch.ones_like(y_pred_colloc_tmp), create_graph=True)[0]
            mu_pred_tmp = -dy_dx_tmp * (y_std / x_std)
            phys_loss_tmp = torch.mean((mu_pred_tmp - model.get_mu()) ** 2)
            phys_loss_tmp.backward(retain_graph=True)
            grads_phys = [p.grad.abs().mean() for p in model.net.parameters() if p.grad is not None]
            mean_grad_phys = torch.mean(torch.stack(grads_phys)) if grads_phys else torch.tensor(1.0).to(device)

            with torch.no_grad():
                lambda_hat = max_grad_data / (mean_grad_phys + 1e-8)
                lambda_phys = alpha_ema * lambda_phys + (1 - alpha_ema) * lambda_hat.item()
                lambda_phys = min(max(lambda_phys, 0.05), 20.0)

        optimizer.zero_grad()

        y_pred_data = model(x_data_t)
        data_loss = torch.mean((y_pred_data - y_data_t) ** 2)

        y_pred_colloc = model(x_colloc_t)
        dy_dx = torch.autograd.grad(y_pred_colloc, x_colloc_t, torch.ones_like(y_pred_colloc), create_graph=True)[0]
        mu_pred = -dy_dx * (y_std / x_std)
        phys_loss = torch.mean((mu_pred - model.get_mu()) ** 2)

        # Stationary metric for checkpointing
        eval_loss = data_loss.item() + phys_loss.item()
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_model_state = copy.deepcopy(model.state_dict())

        loss = data_loss + lambda_phys * phys_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model.get_mu().item()


# ==========================================
# SMOKE TEST RUNNER
# ==========================================
def run_smoke_test():
    for mat_key in MATERIALS_TO_TEST:
        fpath = resolve_file(mat_key)
        if not fpath:
            print(f"[SKIP] {mat_key}: file not found")
            continue

        df = pd.read_csv(fpath)
        x_raw = df["thickness_mm"].values.astype(np.float64)
        y_raw = df["net_counts"].values.astype(np.float64)
        x_max = x_raw.max()
        actual_N = len(x_raw)

        slope, intercept, _, _, _ = linregress(x_raw, np.log(y_raw))
        true_mu = -slope

        print(f"\n{mat_key.capitalize()} | N={actual_N} | true_mu={true_mu:.5f} | noise={NOISE_STD*100:.1f}%")
        print("-" * 75)

        rpes = []
        mat_hash = abs(hash(mat_key)) % 10000
        for trial in range(N_SMOKE_TRIALS):
            np.random.seed(mat_hash + 1000 * actual_N + 100 * 4 + trial)
            torch.manual_seed(mat_hash + 1000 * actual_N + 100 * 4 + trial)

            # Unbiased log-normal multiplicative noise: strictly positive
            eps = np.random.normal(0, NOISE_STD, size=actual_N)
            y_noisy = y_raw * np.exp(eps - 0.5 * (NOISE_STD ** 2))

            initial_mu_guess = robust_initial_slope(x_raw, y_noisy)

            p_mu = train_pinn_lhs(x_raw, y_noisy, x_max, initial_mu_guess, trial)
            rpe = abs((p_mu - true_mu) / true_mu) * 100.0
            rpes.append(rpe)
            print(f"  trial {trial:2d}: initial_mu={initial_mu_guess:.5f} -> estimated_mu={p_mu:.5f} | RPE={rpe:6.2f}%")

        rpes = np.array(rpes)
        n_outliers = int(np.sum(rpes > 200.0))
        print(f"\n  SUMMARY: mean={np.mean(rpes):.1f}% | median={np.median(rpes):.1f}% | "
              f"outliers(>200%)={n_outliers}/{N_SMOKE_TRIALS}")
        print("=" * 75)


if __name__ == "__main__":
    run_smoke_test()