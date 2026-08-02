"""
UNIFIED MASTER ABLATION & VISUALIZATION ENGINE (REAL DATA ONLY)
=============================================================================
1. Executes 30-trial Monte Carlo ablation across PINN, WNLLS, and GP.
2. Uses ONLY real measurement data (no synthetic augmentation).
3. Robust statistics: Tracks outliers (>200% RPE) and reports medians.
4. Three-way Wilcoxon signed-rank tests with explicit failure/exclusion tracking.
5. Dynamic LHS collocation seeding for genuine Monte Carlo variance.
6. Hardened execution with exception handling for classical estimators.
"""

import os
import copy
import warnings
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc, linregress, wilcoxon
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

try:
    import torch_directml
except ImportError:
    torch_directml = None

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION & PARAMETERS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ALT_DATA_DIR = os.path.join(BASE_DIR, "PINN-for-gamma-rays", "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MATERIALS = {
    "aluminum": ("Aluminum", "tab:blue"),
    "copper":   ("Copper",   "tab:orange"),
    "brass":    ("Brass",    "tab:pink"),
    "steel":    ("Steel",    "tab:green"),
    "lead":     ("Lead",     "tab:cyan")
}

SAMPLE_SIZES = [5, 'ALL']
NOISE_TIERS = [0.0, 0.0316, 0.10, 0.3162, 0.5623]
N_TRIALS = 30
PINN_EPOCHS = 3000
M_COLLOC = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu" and torch_directml is not None:
    device = torch_directml.device()

TORCH_DTYPE = torch.float32 

print(f"Executing Master Ablation Suite (Real Data Only) on compute device: {device}")
print("=" * 75)


def resolve_material_file(mat_key):
    candidates = [
        os.path.join(DATA_DIR, f"{mat_key}.csv"),
        os.path.join(ALT_DATA_DIR, f"{mat_key}.csv"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    for folder in (DATA_DIR, ALT_DATA_DIR):
        if os.path.isdir(folder):
            alt_files = [f for f in os.listdir(folder) if mat_key in f.lower() and f.endswith(".csv") and "synthetic" not in f.lower()]
            if alt_files:
                return os.path.join(folder, alt_files[0])

    return None

# ==========================================
# STATISTICAL UTILITIES
# ==========================================
def robust_stats(rpe_array, outlier_thresh=200.0):
    valid = rpe_array[~np.isnan(rpe_array)]
    n_outliers = int(np.sum(valid > outlier_thresh))
    return {
        "mean": np.nanmean(rpe_array) if len(valid) > 0 else np.nan,
        "median": np.nanmedian(rpe_array) if len(valid) > 0 else np.nan,
        "std": np.nanstd(rpe_array) if len(valid) > 0 else np.nan,
        "outliers": n_outliers,
    }

def paired_wilcoxon(a_rpe, b_rpe, label):
    valid = ~np.isnan(a_rpe) & ~np.isnan(b_rpe)
    n_excluded = int(np.sum(~valid))
    if np.sum(valid) >= 5:
        try:
            _, p = wilcoxon(a_rpe[valid], b_rpe[valid], alternative='less')
        except Exception:
            p = np.nan
    else:
        p = np.nan
    return p, n_excluded

# ==========================================
# 1. PINN ARCHITECTURE WITH LHS COLLOCATION
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

def train_pinn_lhs(x_train, y_train, x_max, initial_mu, trial, epochs=PINN_EPOCHS, track_loss=False):
    x_mean, x_std = x_train.mean(), x_train.std()
    y_log = np.log(y_train)
    y_mean, y_std = y_log.mean(), y_log.std()

    x_data_t = torch.tensor((x_train - x_mean) / x_std, dtype=TORCH_DTYPE).view(-1, 1).to(device)
    y_data_t = torch.tensor((y_log - y_mean) / y_std, dtype=TORCH_DTYPE).view(-1, 1).to(device)

    # Dynamic seed for genuine Monte Carlo coverage
    sampler = qmc.LatinHypercube(d=1, seed=42 + trial)
    lhs_samples = sampler.random(n=M_COLLOC) * x_max
    x_colloc_t = torch.tensor((lhs_samples - x_mean) / x_std, dtype=TORCH_DTYPE).view(-1, 1).to(device)
    x_colloc_t.requires_grad_(True)

    model = PINN(initial_mu=initial_mu).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 1. ReduceLROnPlateau tracks loss stagnation dynamically
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=150, min_lr=1e-6)

    history = {'data': [], 'phys': [], 'total': []} if track_loss else None

    # 2. Early stopping structures
    best_loss = float('inf')
    best_model_state = None
    
    # 3. Adaptive Loss Weighting parameters
    lambda_phys = 1.0
    alpha_ema = 0.9  # Exponential moving average momentum

    for epoch in range(epochs):
        # --- ADAPTIVE WEIGHTING CALCULATION (Wang et al. method) ---
        # Evaluate gradients every 10 epochs to reduce computational overhead
        if epoch % 10 == 0:
            # Data Loss gradients
            optimizer.zero_grad()
            y_pred_data_tmp = model(x_data_t)
            data_loss_tmp = torch.mean((y_pred_data_tmp - y_data_t) ** 2)
            data_loss_tmp.backward(retain_graph=True)
            grads_data = [p.grad.abs().max() for p in model.net.parameters() if p.grad is not None]
            max_grad_data = torch.max(torch.stack(grads_data)) if grads_data else torch.tensor(1.0).to(device)
            
            # Physics Loss gradients
            optimizer.zero_grad()
            y_pred_colloc_tmp = model(x_colloc_t)
            dy_dx_tmp = torch.autograd.grad(y_pred_colloc_tmp, x_colloc_t, torch.ones_like(y_pred_colloc_tmp), create_graph=True)[0]
            mu_pred_tmp = -dy_dx_tmp * (y_std / x_std)
            phys_loss_tmp = torch.mean((mu_pred_tmp - torch.exp(model.log_mu)) ** 2)
            phys_loss_tmp.backward(retain_graph=True)
            grads_phys = [p.grad.abs().mean() for p in model.net.parameters() if p.grad is not None]
            mean_grad_phys = torch.mean(torch.stack(grads_phys)) if grads_phys else torch.tensor(1.0).to(device)
            
            # Apply EMA to smooth lambda scaling
            with torch.no_grad():
                lambda_hat = max_grad_data / (mean_grad_phys + 1e-8)
                lambda_phys = alpha_ema * lambda_phys + (1 - alpha_ema) * lambda_hat.item()
                lambda_phys = min(max(lambda_phys, 0.01), 100.0)  # Bounds for stability

        optimizer.zero_grad()
        
        # Standard Forward Pass
        y_pred_data = model(x_data_t)
        data_loss = torch.mean((y_pred_data - y_data_t) ** 2)
        
        y_pred_colloc = model(x_colloc_t)
        dy_dx = torch.autograd.grad(y_pred_colloc, x_colloc_t, torch.ones_like(y_pred_colloc), create_graph=True)[0]
        mu_pred = -dy_dx * (y_std / x_std)
        phys_loss = torch.mean((mu_pred - torch.exp(model.log_mu)) ** 2)
        
        # Combined objective
        loss = data_loss + lambda_phys * phys_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Step the plateau scheduler with the current loss magnitude
        scheduler.step(loss)

        # --- EARLY STOPPING TRIGGER ---
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = copy.deepcopy(model.state_dict())

        if track_loss and (epoch % 10 == 0):
            history['data'].append(data_loss.item())
            history['phys'].append(phys_loss.item())
            history['total'].append(loss.item())

    # Restore minimum-loss weights to prevent utilizing overfit/degraded weights at epoch 3000
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    final_mu = torch.exp(model.log_mu).item()
    return (final_mu, history) if track_loss else final_mu

# ==========================================
# 2. CLASSICAL ESTIMATORS (WNLLS & GP)
# ==========================================
def fit_wnlls(x_train, y_train, eps=1e-8):
    x = np.asarray(x_train, dtype=np.float64).flatten()
    y = np.asarray(y_train, dtype=np.float64).flatten()
    y = np.clip(y, eps, None)

    log_y = np.log(y)
    w = y  

    A = np.column_stack([np.ones_like(x), -x])  
    W = w[:, None]

    ATA = A.T @ (W * A)
    ATb = A.T @ (w * log_y)

    try:
        beta = np.linalg.solve(ATA, ATb)
    except Exception:
        return np.nan, np.nan

    log_I0, mu = beta[0], beta[1]
    if not np.isfinite(mu) or mu <= 0:
        return np.nan, np.nan
    return np.exp(log_I0), mu


def fit_gp(x_train, y_train, x_max):
    try:
        log_y = np.log(np.maximum(y_train, 1e-8))
        kernel = 1.0 * RBF(length_scale=5.0, length_scale_bounds=(1e-2, 1e4)) + \
                 WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-8, 10.0))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=0)
        gp.fit(x_train.reshape(-1, 1), log_y)
        x_dense = np.linspace(0, x_max, 500).reshape(-1, 1)
        log_y_pred = gp.predict(x_dense)
        dx = x_dense[1, 0] - x_dense[0, 0]
        mu_pred = -np.mean(np.gradient(log_y_pred, dx))
        return gp, mu_pred
    except Exception:
        return None, np.nan

# ==========================================
# 3. VISUALIZATION EXPORTERS
# ==========================================
def plot_loss_curves(history, material_name):
    epochs = np.arange(0, PINN_EPOCHS, 10)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history['data'], color='tab:blue')
    axes[0].set_title(r"Data Loss ($\mathcal{L}_{data}$)")
    axes[0].set_yscale('log')

    axes[1].plot(epochs, history['phys'], color='tab:red')
    axes[1].set_title(r"Physics Loss ($\mathcal{L}_{phys}$)")
    axes[1].set_yscale('log')

    axes[2].plot(epochs, history['total'], color='tab:green')
    axes[2].set_title(r"Total Loss ($\mathcal{L}_{total}$)")
    axes[2].set_yscale('log')

    for ax in axes:
        ax.set_xlabel("Epochs")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"{material_name} - PINN Convergence Dynamics", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{material_name.lower()}_loss_curves.png"), dpi=300)
    plt.close()


def plot_fit_comparisons(x_data, y_noisy, x_max, true_mu, true_I0,
                         pinn_mu, pinn_I0, wnlls_mu, wnlls_I0, gp_model, material_name):
    x_dense = np.linspace(0, x_max, 200)

    y_true = true_I0 * np.exp(-true_mu * x_dense)
    y_pinn = pinn_I0 * np.exp(-pinn_mu * x_dense)

    plt.figure(figsize=(8, 6))
    plt.scatter(x_data, y_noisy, color='black', label='Noisy Observations', zorder=5, alpha=0.6)

    plt.plot(x_dense, y_true, 'k--', linewidth=2, label=rf'True Physics ($\mu$={true_mu:.4f})')
    plt.plot(x_dense, y_pinn, color='tab:red', linewidth=2, label=rf'PINN Fit ($\mu$={pinn_mu:.4f})')
    
    if np.isfinite(wnlls_mu) and np.isfinite(wnlls_I0):
        y_wnlls = wnlls_I0 * np.exp(-wnlls_mu * x_dense)
        plt.plot(x_dense, y_wnlls, color='tab:blue', linestyle='-.', linewidth=2,
                  label=rf'WNLLS Fit ($\mu$={wnlls_mu:.4f})')
    
    if gp_model is not None:
        y_gp = np.exp(gp_model.predict(x_dense.reshape(-1, 1)))
        plt.plot(x_dense, y_gp, color='tab:green', linestyle=':', linewidth=2, label='GP Mean Fit')

    plt.yscale('log')
    plt.title(f"{material_name} - Algorithmic Curve Fits (High Noise Regime)", fontweight='bold')
    plt.xlabel("Thickness (mm)")
    plt.ylabel("Photon Counts (Log Scale)")
    plt.legend(framealpha=0.9)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{material_name.lower()}_curve_fits.png"), dpi=300)
    plt.close()


def generate_figure2_panel(summary_df):
    print("\nGenerating Figure 2 (1x2 Robustness Panel based on MEDIANS)...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for i, n_val in enumerate(SAMPLE_SIZES):
        ax = axes[i]
        subset = summary_df[summary_df['N_label'] == str(n_val)]

        for algo, style, color, label in [
            ('pinn',  '-o', 'tab:red',   'PINN (Physics-Informed)'),
            ('wnlls', '--s', 'tab:blue',  'WNLLS (Closed-Form Log-Linear)'),
            ('gp',    ':^',  'tab:green', 'GP Regressor')
        ]:
            # Plotting medians so GP outliers don't destroy the visual scale
            medians = subset.groupby('noise_tier')[f'{algo}_rpe_median'].mean()
            ax.plot(NOISE_TIERS, medians, style, color=color, label=label, linewidth=1.8, markersize=6)

        title_str = f"Sample Size: N = {n_val}" if n_val != 'ALL' else "Sample Size: N = Full Dataset"
        ax.set_title(title_str, fontsize=12, fontweight='bold')
        ax.set_xlabel(r"Noise Standard Deviation ($\sigma_{noise}$)", fontsize=10)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel("Median Relative Parameter Error (%)", fontsize=10)

    axes[0].legend(loc='upper left', fontsize=9, framealpha=0.9)
    plt.suptitle("Algorithmic Robustness Across Sampling Density and Noise Regimes",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig_path = os.path.join(RESULTS_DIR, "figure2_master_robustness.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Figure 2 exported to: {fig_path}")

# ==========================================
# 4. MASTER EXECUTION ENGINE
# ==========================================
def run_master_suite():
    all_results = []

    for mat_key, (mat_name, color) in MATERIALS.items():
        fpath = resolve_material_file(mat_key)
        if not fpath:
            print(f"[WARNING] Skipping {mat_name}: No real CSV found in {DATA_DIR} or {ALT_DATA_DIR}")
            continue

        df = pd.read_csv(fpath)
        x_raw = df["thickness_mm"].values.astype(np.float64)
        y_raw = df["net_counts"].values.astype(np.float64)
        x_max = x_raw.max()

        slope, intercept, _, _, _ = linregress(x_raw, np.log(y_raw))
        true_mu = -slope
        clean_I0 = np.exp(intercept)

        print(f"\nEvaluating: {mat_name:<10} | Source File: {os.path.basename(fpath)} | Total Points: {len(x_raw)}")
        print(f"Derived True mu: {true_mu:.5f} mm^-1 | Derived I0: {clean_I0:.0f} counts")
        print("-" * 75)

        evaluated_Ns = set()
        for N_req in SAMPLE_SIZES:
            actual_N = len(x_raw) if N_req == 'ALL' else min(N_req, len(x_raw))
            if actual_N in evaluated_Ns:
                continue
            evaluated_Ns.add(actual_N)

            for noise_idx, noise_std in enumerate(NOISE_TIERS):
                pinn_mus, wnlls_mus, gp_mus = [], [], []

                for trial in range(N_TRIALS):
                    np.random.seed(1000 * actual_N + 100 * noise_idx + trial)
                    torch.manual_seed(1000 * actual_N + 100 * noise_idx + trial)

                    idx = np.linspace(0, len(x_raw) - 1, actual_N, dtype=int)
                    x_sub, y_sub = x_raw[idx], y_raw[idx]

                    if noise_std > 0:
                        noise_arr = np.random.normal(0, noise_std, size=actual_N)
                        y_noisy = np.clip(y_sub * (1 + noise_arr), a_min=1e-5, a_max=None)
                    else:
                        y_noisy = y_sub

                    slope_guess, _, _, _, _ = linregress(x_sub, np.log(y_noisy))
                    initial_mu_guess = max(-slope_guess, 1e-4)

                    track_visuals = (trial == 0 and N_req == 'ALL' and noise_idx == len(NOISE_TIERS) - 1)

                    if track_visuals:
                        p_mu, history = train_pinn_lhs(x_sub, y_noisy, x_max, initial_mu_guess, trial, track_loss=True)
                        plot_loss_curves(history, mat_name)
                    else:
                        p_mu = train_pinn_lhs(x_sub, y_noisy, x_max, initial_mu_guess, trial)

                    w_I0, w_mu = fit_wnlls(x_sub, y_noisy)
                    gp_mod, g_mu = fit_gp(x_sub, y_noisy, x_max)

                    if track_visuals:
                        p_I0 = np.exp(np.mean(np.log(y_noisy) + p_mu * x_sub))
                        plot_fit_comparisons(x_sub, y_noisy, x_max, true_mu, clean_I0,
                                             p_mu, p_I0, w_mu, w_I0, gp_mod, mat_name)

                    pinn_mus.append(p_mu)
                    wnlls_mus.append(w_mu)
                    gp_mus.append(g_mu)

                wnlls_failures = int(np.isnan(wnlls_mus).sum())

                pinn_rpe = np.abs((np.array(pinn_mus) - true_mu) / true_mu) * 100.0
                wnlls_rpe = np.abs((np.array(wnlls_mus) - true_mu) / true_mu) * 100.0
                gp_rpe = np.abs((np.array(gp_mus) - true_mu) / true_mu) * 100.0
                
                # Apply robust statistics
                pinn_stats = robust_stats(pinn_rpe)
                wnlls_stats = robust_stats(wnlls_rpe)
                gp_stats = robust_stats(gp_rpe)

                # Three-way Wilcoxon Signed-Rank Tests
                p_wnlls_vs_pinn, excl_w_p = paired_wilcoxon(wnlls_rpe, pinn_rpe, "WNLLS<PINN")
                p_gp_vs_pinn, excl_g_p = paired_wilcoxon(gp_rpe, pinn_rpe, "GP<PINN")
                p_wnlls_vs_gp, excl_w_g = paired_wilcoxon(wnlls_rpe, gp_rpe, "WNLLS<GP")

                all_results.append({
                    "material": mat_name, 
                    "N_actual": actual_N, 
                    "N_label": str(N_req),
                    "noise_tier": noise_idx, 
                    "noise_std": noise_std,
                    "pinn_rpe_mean": pinn_stats["mean"], 
                    "pinn_rpe_median": pinn_stats["median"],
                    "pinn_rpe_std": pinn_stats["std"], 
                    "pinn_outliers": pinn_stats["outliers"],
                    "wnlls_rpe_mean": wnlls_stats["mean"], 
                    "wnlls_rpe_median": wnlls_stats["median"],
                    "wnlls_rpe_std": wnlls_stats["std"], 
                    "wnlls_failures": wnlls_failures,
                    "wnlls_outliers": wnlls_stats["outliers"],
                    "gp_rpe_mean": gp_stats["mean"], 
                    "gp_rpe_median": gp_stats["median"],
                    "gp_rpe_std": gp_stats["std"], 
                    "gp_outliers": gp_stats["outliers"],
                    "pval_wnlls_vs_pinn": p_wnlls_vs_pinn, 
                    "excl_wnlls_vs_pinn": excl_w_p,
                    "pval_gp_vs_pinn": p_gp_vs_pinn,       
                    "excl_gp_vs_pinn": excl_g_p,
                    "pval_wnlls_vs_gp": p_wnlls_vs_gp,     
                    "excl_wnlls_vs_gp": excl_w_g
                })

                pd.DataFrame(all_results).to_csv(
                    os.path.join(RESULTS_DIR, "table1_master_robustness_LIVE.csv"), index=False)

                fail_str = f" (WNLLS Fails: {wnlls_failures}/{N_TRIALS})" if wnlls_failures > 0 else ""
                gp_out_str = f" (GP Outliers: {gp_stats['outliers']})" if gp_stats['outliers'] > 0 else ""
                
                print(f"N={actual_N:3d} | SNR Tier {noise_idx} ({noise_std*100:4.1f}%) -> MEDIANS: "
                      f"PINN: {pinn_stats['median']:5.1f}% | "
                      f"WNLLS: {wnlls_stats['median']:5.1f}%{fail_str} | "
                      f"GP: {gp_stats['median']:5.1f}%{gp_out_str}")

    if all_results:
        summary_df = pd.DataFrame(all_results)
        csv_path = os.path.join(RESULTS_DIR, "table1_master_robustness.csv")
        summary_df.to_csv(csv_path, index=False)
        print("\n" + "=" * 75)
        print(f"[SUCCESS] Master robustness table (with median/outlier tracking) exported to: {csv_path}")

        generate_figure2_panel(summary_df)
        print("=" * 75)
        print("[ALL TASKS COMPLETED SUCCESSFULLY]")

if __name__ == "__main__":
    run_master_suite()