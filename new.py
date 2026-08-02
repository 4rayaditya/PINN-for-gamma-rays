"""
SMOKE TEST — patched PINN only, on the 3 materials that showed 5/30 outliers
=============================================================================
Run this BEFORE the full ablation suite. It only touches PINN (WNLLS/GP are
unchanged and already validated), only tests aluminum/copper/brass, only at
N=full and the worst noise tier (56.2%), and only 10 trials instead of 30.
That's ~1/45th the compute of a full rerun, but it isolates exactly the
cells that mattered.

Reads real CSVs from the same DATA_DIR your main script uses — point
DATA_DIR at wherever aluminum.csv / copper.csv / brass.csv actually live.

What to look at in the output:
  - "outliers (RPE>200%)" should drop from 5/10 (scaled-down expectation
    from the old 5/30) toward 0-1/10. If it's still 4-5/10, the fix isn't
    working and the full rerun isn't worth doing yet.
  - "median" should stay in a similar ballpark to before (~90-98% in the
    old runs) — a fix that also wrecks the median accuracy isn't a real fix,
    just a different failure mode.
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

# ==========================================
# CONFIG — EDIT THIS PATH
# ==========================================
DATA_DIR = "./data"   # <-- point this at the folder containing aluminum.csv, copper.csv, brass.csv

MATERIALS_TO_TEST = ["aluminum", "copper", "brass"]
NOISE_STD = 0.5623     # the 56.2% tier that showed the outliers
N_SMOKE_TRIALS = 10    # subset of the full 30, just enough to see the pattern
PINN_EPOCHS = 3000
M_COLLOC = 500

# Which training variant to test — run this script 3 times, once per value,
# to isolate which change actually helps vs. hurts:
#   "baseline"        = original: fixed lambda ramp (0.1->5.0), cosine LR, no checkpoint
#   "checkpoint_only"  = baseline schedule/LR + best-weights checkpoint (isolates checkpoint effect)
#   "adaptive_full"    = the full new version: adaptive lambda_phys + ReduceLROnPlateau + checkpoint
VARIANT = "checkpoint_only"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = torch.float32

print(f"Smoke test running on: {device}")
print(f"Variant under test: {VARIANT}")
print("=" * 75)


# ==========================================
# PINN — same architecture as the main script
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
        self.log_mu = nn.Parameter(torch.tensor(np.log(initial_mu), dtype=TORCH_DTYPE))

    def forward(self, x):
        return self.net(x)


def train_pinn_lhs(x_train, y_train, x_max, initial_mu, trial, epochs=PINN_EPOCHS):
    """Three variants controlled by the VARIANT flag above, so the effect of
    each change can be isolated instead of guessed at."""
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

    use_adaptive_lambda = (VARIANT == "adaptive_full")
    use_checkpoint = (VARIANT in ("checkpoint_only", "adaptive_full"))

    if use_adaptive_lambda:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=150, min_lr=1e-6)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_loss = float('inf')
    best_model_state = None
    lambda_phys = 1.0
    alpha_ema = 0.9

    for epoch in range(epochs):
        if use_adaptive_lambda and epoch % 10 == 0:
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
            phys_loss_tmp = torch.mean((mu_pred_tmp - torch.exp(model.log_mu)) ** 2)
            phys_loss_tmp.backward(retain_graph=True)
            grads_phys = [p.grad.abs().mean() for p in model.net.parameters() if p.grad is not None]
            mean_grad_phys = torch.mean(torch.stack(grads_phys)) if grads_phys else torch.tensor(1.0).to(device)

            with torch.no_grad():
                lambda_hat = max_grad_data / (mean_grad_phys + 1e-8)
                lambda_phys = alpha_ema * lambda_phys + (1 - alpha_ema) * lambda_hat.item()
                lambda_phys = min(max(lambda_phys, 0.01), 100.0)
        elif not use_adaptive_lambda:
            # original fixed ramp schedule, 0.1 -> 5.0 over first half of training
            lambda_phys = min(5.0, 0.1 + (5.0 - 0.1) * epoch / (epochs // 2))

        optimizer.zero_grad()

        y_pred_data = model(x_data_t)
        data_loss = torch.mean((y_pred_data - y_data_t) ** 2)

        y_pred_colloc = model(x_colloc_t)
        dy_dx = torch.autograd.grad(y_pred_colloc, x_colloc_t, torch.ones_like(y_pred_colloc), create_graph=True)[0]
        mu_pred = -dy_dx * (y_std / x_std)
        phys_loss = torch.mean((mu_pred - torch.exp(model.log_mu)) ** 2)

        loss = data_loss + lambda_phys * phys_loss

        if use_checkpoint:
            # FIX: checkpoint BEFORE backward()/step(), so best_model_state
            # actually matches the loss value that selected it
            current_loss = loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                best_model_state = copy.deepcopy(model.state_dict())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if use_adaptive_lambda:
            scheduler.step(loss)
        else:
            scheduler.step()

    if use_checkpoint and best_model_state is not None:
        model.load_state_dict(best_model_state)

    return torch.exp(model.log_mu).item()


# ==========================================
# SMOKE TEST LOOP
# ==========================================
def run_smoke_test():
    for mat_key in MATERIALS_TO_TEST:
        fpath = os.path.join(DATA_DIR, f"{mat_key}.csv")
        if not os.path.exists(fpath):
            print(f"[SKIP] {mat_key}: file not found at {fpath}")
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
            # material hash added so same-N materials (e.g. copper/brass, both N=6)
            # don't get bit-identical noise/init/collocation every trial
            np.random.seed(mat_hash + 1000 * actual_N + 100 * 4 + trial)
            torch.manual_seed(mat_hash + 1000 * actual_N + 100 * 4 + trial)

            noise_arr = np.random.normal(0, NOISE_STD, size=actual_N)
            y_noisy = np.clip(y_raw * (1 + noise_arr), a_min=1e-5, a_max=None)

            slope_guess, _, _, _, _ = linregress(x_raw, np.log(y_noisy))
            initial_mu_guess = max(-slope_guess, 1e-4)

            p_mu = train_pinn_lhs(x_raw, y_noisy, x_max, initial_mu_guess, trial)
            rpe = abs((p_mu - true_mu) / true_mu) * 100.0
            rpes.append(rpe)
            print(f"  trial {trial:2d}: mu={p_mu:.5f}  RPE={rpe:8.2f}%")

        rpes = np.array(rpes)
        n_outliers = int(np.sum(rpes > 200.0))
        print(f"\n  SUMMARY: mean={np.mean(rpes):.1f}%  median={np.median(rpes):.1f}%  "
              f"outliers(>200%)={n_outliers}/{N_SMOKE_TRIALS}")
        print("=" * 75)


if __name__ == "__main__":
    run_smoke_test()