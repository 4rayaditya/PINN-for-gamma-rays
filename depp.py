"""
DEEP DIAGNOSTIC — trace log_mu trajectory epoch-by-epoch for known-bad seeds
=============================================================================
Purpose: confirm or refute the hypothesis that mu = exp(log_mu) traps the
optimizer once log_mu drifts far enough in either direction, because
d(exp(log_mu))/d(log_mu) = exp(log_mu), which vanishes as log_mu -> -inf
and explodes as log_mu -> +inf (making large steps in the wrong direction
easy to trigger and hard to correct either way).

This targets two specific reproduced cases from your checkpoint_only run:
  - Aluminum, trial 9  -> collapsed toward the low tail (final mu=0.15331,
    RPE=814% — actually an EXPLOSION case, kept as a second explosion
    example alongside Copper trial 0)
  - Aluminum, trial 4  -> COLLAPSE case (final mu=0.00008, RPE=99.54%)
  - Copper, trial 0    -> EXPLOSION case (final mu=0.58727, RPE=1126.61%)

Logs every 25 epochs: log_mu, mu, |grad of log_mu|, data_loss, phys_loss,
lambda_phys. Look for:
  - log_mu drifting to an extreme value EARLY (well before epoch 3000) and
    then flatlining for the rest of training
  - |grad of log_mu| shrinking toward ~0 at the same point log_mu flatlines
    (confirms the vanishing-gradient trap)
  - OR: if log_mu keeps moving throughout training with no early flatline,
    the trap hypothesis is wrong and something else is going on — report
    back either way, don't assume the fix before seeing this
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

# --- determinism: without this, CUDA ops can execute in different
# accumulation order run-to-run even with matching torch.manual_seed(),
# which is almost certainly why trial 9 and Copper trial 0 gave completely
# different outcomes here than in the original smoke test run ---
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

DATA_DIR = "./data"   # <-- same as before, point at your real CSVs

# (material_key, trial_number) pairs to trace, pulled directly from your
# checkpoint_only run's worst cases
CASES_TO_TRACE = [
    ("aluminum", 4),   # collapse case: final mu=0.00008
    ("aluminum", 9),   # explosion case: final mu=0.15331
    ("copper", 0),      # explosion case: final mu=0.58727
]

NOISE_STD = 0.5623
PINN_EPOCHS = 3000
M_COLLOC = 500
LOG_EVERY = 25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = torch.float32

print(f"Deep diagnostic running on: {device}")
print("=" * 90)


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


def train_pinn_traced(x_train, y_train, x_max, initial_mu, trial, epochs=PINN_EPOCHS):
    """Same as the checkpoint_only variant, but logs log_mu / its gradient /
    both loss terms every LOG_EVERY epochs instead of only returning the
    final value."""
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_loss = float('inf')
    best_model_state = None

    trace = []

    for epoch in range(epochs):
        lambda_phys = min(5.0, 0.1 + (5.0 - 0.1) * epoch / (epochs // 2))

        optimizer.zero_grad()

        y_pred_data = model(x_data_t)
        data_loss = torch.mean((y_pred_data - y_data_t) ** 2)

        y_pred_colloc = model(x_colloc_t)
        dy_dx = torch.autograd.grad(y_pred_colloc, x_colloc_t, torch.ones_like(y_pred_colloc), create_graph=True)[0]
        mu_pred = -dy_dx * (y_std / x_std)
        phys_loss = torch.mean((mu_pred - torch.exp(model.log_mu)) ** 2)

        loss = data_loss + lambda_phys * phys_loss

        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_model_state = copy.deepcopy(model.state_dict())

        loss.backward()

        if epoch % LOG_EVERY == 0:
            grad_log_mu = model.log_mu.grad.item() if model.log_mu.grad is not None else float('nan')
            trace.append({
                "epoch": epoch,
                "log_mu": model.log_mu.item(),
                "mu": torch.exp(model.log_mu).item(),
                "grad_log_mu": grad_log_mu,
                "data_loss": data_loss.item(),
                "phys_loss": phys_loss.item(),
                "lambda_phys": lambda_phys,
            })

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    final_mu = torch.exp(model.log_mu).item()
    return final_mu, pd.DataFrame(trace)


def run_diagnostic():
    for mat_key, target_trial in CASES_TO_TRACE:
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

        # NOTE: matches the checkpoint_only run's seeding, i.e. WITHOUT the
        # material-hash fix, since we're deliberately reproducing those
        # exact reported cases
        np.random.seed(1000 * actual_N + 100 * 4 + target_trial)
        torch.manual_seed(1000 * actual_N + 100 * 4 + target_trial)

        noise_arr = np.random.normal(0, NOISE_STD, size=actual_N)
        y_noisy = np.clip(y_raw * (1 + noise_arr), a_min=1e-5, a_max=None)

        slope_guess, _, _, _, _ = linregress(x_raw, np.log(y_noisy))
        initial_mu_guess = max(-slope_guess, 1e-4)

        print(f"\n{mat_key.capitalize()} trial {target_trial} | N={actual_N} | true_mu={true_mu:.5f} | "
              f"initial_mu_guess={initial_mu_guess:.5f}")
        print("-" * 90)

        final_mu, trace_df = train_pinn_traced(x_raw, y_noisy, x_max, initial_mu_guess, target_trial)

        print(f"{'epoch':>6} | {'log_mu':>10} | {'mu':>12} | {'|grad_log_mu|':>14} | "
              f"{'data_loss':>10} | {'phys_loss':>10} | {'lambda':>7}")
        for _, r in trace_df.iterrows():
            print(f"{int(r.epoch):6d} | {r.log_mu:10.4f} | {r.mu:12.6f} | {abs(r.grad_log_mu):14.6e} | "
                  f"{r.data_loss:10.4f} | {r.phys_loss:10.4f} | {r.lambda_phys:7.3f}")

        final_rpe = abs((final_mu - true_mu) / true_mu) * 100
        print(f"\n  FINAL: mu={final_mu:.5f}  RPE={final_rpe:.2f}%")

        # quick automated check: did log_mu stop moving substantially in the
        # back half of training while the gradient also shrank?
        half = len(trace_df) // 2
        early_log_mu_range = trace_df["log_mu"].iloc[:half].max() - trace_df["log_mu"].iloc[:half].min()
        late_log_mu_range = trace_df["log_mu"].iloc[half:].max() - trace_df["log_mu"].iloc[half:].min()
        late_grad_mean = trace_df["grad_log_mu"].iloc[half:].abs().mean()
        print(f"  log_mu range, first half of training: {early_log_mu_range:.4f}")
        print(f"  log_mu range, second half of training: {late_log_mu_range:.4f}")
        print(f"  mean |grad_log_mu|, second half: {late_grad_mean:.6e}")
        if late_log_mu_range < 0.05 and late_grad_mean < 1e-3:
            print("  -> CONSISTENT WITH TRAP HYPOTHESIS: log_mu flatlined with a near-vanished gradient.")
        else:
            print("  -> NOT clearly consistent with the trap hypothesis — log_mu kept moving or gradient stayed live.")

        trace_df.to_csv(f"trace_{mat_key}_trial{target_trial}.csv", index=False)
        print(f"  (full trace saved to trace_{mat_key}_trial{target_trial}.csv)")
        print("=" * 90)


if __name__ == "__main__":
    run_diagnostic()