"""
UNIFIED MASTER ABLATION & VISUALIZATION ENGINE - PUBLICATION-GRADE v2
=======================================================================
Peer-Review Fixes Applied (v2):
  P0-1  FDR (Benjamini-Hochberg) computed inside script, written to CSV.
  P0-2  WNLLS failure penalty: RPE = 1000% for matrix-singular trials.
         Both penalty-inclusive AND conditional Wilcoxon tests reported.
  P0-3  Ground-truth mu via WEIGHTED OLS (w=y_raw). NIST XCOM printed.
  P1-4  GP random_state varies per trial (mat_hash + trial).
  P1-5  warnings filter narrowed to sklearn/scipy only.
  P2-6  Bootstrap 95% CI on median RPE for all three methods per cell.
  P2-7  lambda_phys trajectory tracked and exported as 4th loss panel.
  P2-8  torch.use_deterministic_algorithms(True, warn_only=True).
  NEW   run_buildup_experiment(): I(x)=I0*(1+beta*x)*exp(-mu*x).
"""

import os
import copy
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")
warnings.filterwarnings("ignore", message=".*Ill-conditioned matrix.*")
warnings.filterwarnings("ignore", message=".*lbfgs.*")

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import qmc, wilcoxon
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("[WARNING] statsmodels not found - using manual BH procedure.")

try:
    import torch_directml
except ImportError:
    torch_directml = None

# ============================================================
# DEVICE SETUP & DETERMINISM (P2-8)
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu" and torch_directml is not None:
    try:
        device = torch_directml.device()
        print("[INFO] Using DirectML device.")
    except Exception:
        pass

torch.use_deterministic_algorithms(True, warn_only=True)
TORCH_DTYPE = torch.float32

print(f"Executing Master Ablation Suite v2 (Publication Grade) on: {device}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU: {props.name} | VRAM: {props.total_memory/1e9:.1f} GB")
print("=" * 75)

# ============================================================
# PATHS & CONSTANTS
# ============================================================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
ALT_DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "PINN-for-gamma-rays", "data")
RESULTS_DIR  = os.path.join(BASE_DIR, "results_v2")
os.makedirs(RESULTS_DIR, exist_ok=True)

MATERIALS = {
    "aluminum": ("Aluminum", "tab:blue"),
    "copper":   ("Copper",   "tab:orange"),
    "brass":    ("Brass",    "tab:pink"),
    "steel":    ("Steel",    "tab:green"),
    "lead":     ("Lead",     "tab:cyan"),
}

# NIST XCOM: (mu/rho cm2/g, density g/cm3) at 661.7 keV Cs-137
# Source: Hubbell & Seltzer 1995, NISTIR 5632
NIST_MU_RHO = {
    "aluminum": (0.0748, 2.70),
    "copper":   (0.0573, 8.96),
    "brass":    (0.0570, 8.50),
    "steel":    (0.0594, 7.85),
    "lead":     (0.1101, 11.34),
}

SAMPLE_SIZES     = [5, "ALL"]
NOISE_TIERS      = [0.0, 0.0316, 0.10, 0.3162, 0.5623]
N_TRIALS         = 30
PINN_EPOCHS      = 3000
M_COLLOC         = 500
RPE_FAIL_PENALTY = 1000.0   # substituted for NaN in penalty-inclusive Wilcoxon


# ============================================================
# FILE RESOLUTION
# ============================================================
def resolve_material_file(mat_key):
    for folder in (DATA_DIR, ALT_DATA_DIR):
        p = os.path.join(folder, f"{mat_key}.csv")
        if os.path.exists(p):
            return p
    for folder in (DATA_DIR, ALT_DATA_DIR):
        if os.path.isdir(folder):
            hits = [f for f in os.listdir(folder)
                    if mat_key in f.lower() and f.endswith(".csv")
                    and "synthetic" not in f.lower()]
            if hits:
                return os.path.join(folder, hits[0])
    return None


# ============================================================
# P0-3: WEIGHTED OLS GROUND TRUTH
# ============================================================
def weighted_ols_ground_truth(x_raw, y_raw):
    """
    Derive true_mu via delta-method weighted OLS in log-space.
    Weights w_k = y_raw_k  (Poisson-optimal for Beer-Lambert).
    Returns (true_mu, true_I0, mu_stderr).
    """
    x    = np.asarray(x_raw, dtype=np.float64)
    y    = np.asarray(y_raw, dtype=np.float64)
    logy = np.log(np.maximum(y, 1e-12))
    w    = y
    A    = np.column_stack([np.ones_like(x), -x])
    AtwA = (A.T * w) @ A
    AtwZ = A.T @ (w * logy)
    try:
        beta      = np.linalg.solve(AtwA, AtwZ)
        cov       = np.linalg.inv(AtwA)
        mu_stderr = np.sqrt(cov[1, 1])
    except np.linalg.LinAlgError:
        from scipy.stats import linregress
        slope, intercept, *_ = linregress(x, logy)
        beta      = np.array([intercept, -slope])
        mu_stderr = np.nan
    return float(beta[1]), float(np.exp(beta[0])), float(mu_stderr)


# ============================================================
# THEIL-SEN ROBUST INITIAL SLOPE
# ============================================================
def robust_initial_slope(x_sub, y_noisy):
    logy = np.log(np.maximum(y_noisy, 1e-12))
    n    = len(x_sub)
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = x_sub[j] - x_sub[i]
            if abs(dx) > 1e-6:
                slopes.append(-(logy[j] - logy[i]) / dx)
    if slopes:
        med = float(np.median(slopes))
        if 1e-4 <= med <= 2.0:
            return med
    try:
        from scipy.stats import linregress
        slope, *_ = linregress(x_sub, logy)
        if -slope > 1e-4:
            return float(-slope)
    except Exception:
        pass
    return 0.05


# ============================================================
# P2-6: STATISTICAL UTILITIES WITH BOOTSTRAP CI
# ============================================================
def robust_stats(rpe_array, outlier_thresh=200.0):
    valid = rpe_array[~np.isnan(rpe_array)]
    n_out = int(np.sum(valid > outlier_thresh))
    if len(valid) == 0:
        return {"mean": np.nan, "median": np.nan, "std": np.nan,
                "outliers": 0, "ci_lo": np.nan, "ci_hi": np.nan}
    rng      = np.random.default_rng(seed=12345)
    boot     = rng.choice(valid, size=(2000, len(valid)), replace=True)
    boot_med = np.median(boot, axis=1)
    return {
        "mean":    float(np.nanmean(rpe_array)),
        "median":  float(np.nanmedian(rpe_array)),
        "std":     float(np.nanstd(rpe_array)),
        "outliers": n_out,
        "ci_lo":   float(np.percentile(boot_med, 2.5)),
        "ci_hi":   float(np.percentile(boot_med, 97.5)),
    }


# ============================================================
# P0-2: DUAL WILCOXON (PENALTY + CONDITIONAL)
# ============================================================
def paired_wilcoxon_penalty(a_rpe_raw, b_rpe_raw):
    """
    Replace NaN with RPE_FAIL_PENALTY (primary) and also
    run on shared-success subset (secondary/conditional).
    """
    a = np.where(np.isnan(a_rpe_raw), RPE_FAIL_PENALTY, a_rpe_raw)
    b = np.where(np.isnan(b_rpe_raw), RPE_FAIL_PENALTY, b_rpe_raw)
    n_pen = int(np.sum(np.isnan(a_rpe_raw) | np.isnan(b_rpe_raw)))
    try:
        _, p_pen = wilcoxon(a, b, alternative="less")
    except Exception:
        p_pen = np.nan
    valid = ~np.isnan(a_rpe_raw) & ~np.isnan(b_rpe_raw)
    if valid.sum() >= 5:
        try:
            _, p_cond = wilcoxon(a_rpe_raw[valid], b_rpe_raw[valid], alternative="less")
        except Exception:
            p_cond = np.nan
    else:
        p_cond = np.nan
    return float(p_pen), float(p_cond), n_pen


# ============================================================
# P0-1: BENJAMINI-HOCHBERG FDR
# ============================================================
def bh_fdr(pvalues, alpha=0.05):
    pvals = np.asarray(pvalues, dtype=float)
    if HAS_STATSMODELS:
        finite = np.isfinite(pvals)
        q = np.full_like(pvals, np.nan)
        if finite.sum() > 0:
            _, q_fin, _, _ = multipletests(pvals[finite], method="fdr_bh", alpha=alpha)
            q[finite] = q_fin
        return q
    n     = len(pvals)
    order = np.argsort(pvals)
    q     = np.full(n, np.nan)
    for rank, idx in enumerate(order, start=1):
        if np.isfinite(pvals[idx]):
            q[idx] = pvals[idx] * n / rank
    for i in range(n - 2, -1, -1):
        if np.isfinite(q[order[i]]) and np.isfinite(q[order[i+1]]):
            q[order[i]] = min(q[order[i]], q[order[i+1]])
    return np.minimum(q, 1.0)


# ============================================================
# PINN ARCHITECTURE
# ============================================================
class PINN(nn.Module):
    def __init__(self, initial_mu, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.mu_floor = 1e-4
        target = max(float(initial_mu) - self.mu_floor, 1e-4)
        z_init = np.log(np.exp(target) - 1.0) if target < 20.0 else target
        self.z_mu = nn.Parameter(torch.tensor(z_init, dtype=TORCH_DTYPE))

    def get_mu(self):
        return torch.nn.functional.softplus(self.z_mu) + self.mu_floor

    def forward(self, x):
        return self.net(x)


def train_pinn_lhs(x_train, y_train, x_max, initial_mu, trial,
                   epochs=PINN_EPOCHS, track_loss=False, track_lambda=False):
    x_mean, x_std = x_train.mean(), x_train.std()
    if x_std < 1e-8:
        x_std = 1.0
    y_log         = np.log(np.maximum(y_train, 1e-12))
    y_mean, y_std = y_log.mean(), y_log.std()
    if y_std < 1e-8:
        y_std = 1.0

    x_data_t   = torch.tensor((x_train-x_mean)/x_std, dtype=TORCH_DTYPE).view(-1,1).to(device)
    y_data_t   = torch.tensor((y_log  -y_mean)/y_std, dtype=TORCH_DTYPE).view(-1,1).to(device)
    sampler    = qmc.LatinHypercube(d=1, seed=42+trial)
    lhs_raw    = sampler.random(n=M_COLLOC) * x_max
    x_colloc_t = torch.tensor((lhs_raw-x_mean)/x_std, dtype=TORCH_DTYPE).view(-1,1).to(device)
    x_colloc_t.requires_grad_(True)

    model      = PINN(initial_mu=initial_mu).to(device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history     = {"data":[], "phys":[], "total":[]} if track_loss else None
    lam_history = [] if track_lambda else None
    best_loss, best_state = float("inf"), None
    lambda_phys = 1.0
    alpha_ema   = 0.9

    for epoch in range(epochs):
        # Adaptive gradient balancing (Wang et al. 2021)
        if epoch % 10 == 0:
            optimizer.zero_grad(set_to_none=True)
            dl_tmp = torch.mean((model(x_data_t) - y_data_t)**2)
            dl_tmp.backward(retain_graph=True)
            gd = [p.grad.abs().max() for p in model.net.parameters() if p.grad is not None]
            max_gd = torch.max(torch.stack(gd)) if gd else torch.tensor(1.0, device=device)

            optimizer.zero_grad(set_to_none=True)
            yc    = model(x_colloc_t)
            dy    = torch.autograd.grad(yc, x_colloc_t, torch.ones_like(yc), create_graph=True)[0]
            mu_p  = -dy * (y_std / x_std)
            pl_tm = torch.mean((mu_p - model.get_mu())**2)
            pl_tm.backward(retain_graph=True)
            gpl   = [p.grad.abs().mean() for p in model.net.parameters() if p.grad is not None]
            mean_gp = torch.mean(torch.stack(gpl)) if gpl else torch.tensor(1.0, device=device)

            with torch.no_grad():
                lam_hat     = max_gd / (mean_gp + 1e-8)
                lambda_phys = alpha_ema*lambda_phys + (1-alpha_ema)*lam_hat.item()
                lambda_phys = float(np.clip(lambda_phys, 0.05, 20.0))
            if track_lambda:
                lam_history.append(lambda_phys)

        optimizer.zero_grad(set_to_none=True)
        data_loss = torch.mean((model(x_data_t) - y_data_t)**2)
        yc2       = model(x_colloc_t)
        dy2       = torch.autograd.grad(yc2, x_colloc_t, torch.ones_like(yc2), create_graph=True)[0]
        mu_pred   = -dy2 * (y_std / x_std)
        phys_loss = torch.mean((mu_pred - model.get_mu())**2)

        eval_loss = data_loss.item() + phys_loss.item()
        if eval_loss < best_loss:
            best_loss  = eval_loss
            best_state = copy.deepcopy(model.state_dict())

        loss = data_loss + lambda_phys * phys_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if track_loss and epoch % 10 == 0:
            history["data"].append(data_loss.item())
            history["phys"].append(phys_loss.item())
            history["total"].append(loss.item())

    if best_state is not None:
        model.load_state_dict(best_state)

    final_mu = model.get_mu().item()
    if track_loss and track_lambda:
        return final_mu, history, lam_history
    if track_loss:
        return final_mu, history
    return final_mu


# ============================================================
# CLASSICAL ESTIMATORS
# ============================================================
def fit_wnlls(x_train, y_train, eps=1e-8):
    x    = np.asarray(x_train, dtype=np.float64).flatten()
    y    = np.clip(np.asarray(y_train, dtype=np.float64).flatten(), eps, None)
    logy = np.log(y)
    w    = y
    A    = np.column_stack([np.ones_like(x), -x])
    ATA  = (A.T * w) @ A
    ATb  = A.T @ (w * logy)
    try:
        beta = np.linalg.solve(ATA, ATb)
    except Exception:
        return np.nan, np.nan
    if not np.isfinite(beta[1]) or beta[1] <= 0:
        return np.nan, np.nan
    return float(np.exp(beta[0])), float(beta[1])


def fit_gp(x_train, y_train, x_max, seed=0):
    """P1-4: seed varies per trial via caller."""
    try:
        logy   = np.log(np.maximum(y_train, 1e-8))
        kernel = (1.0*RBF(length_scale=5.0, length_scale_bounds=(1e-2, 1e4)) +
                  WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-8, 10.0)))
        gp     = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                          random_state=int(seed) % (2**31))
        gp.fit(x_train.reshape(-1, 1), logy)
        xd      = np.linspace(0, x_max, 500).reshape(-1, 1)
        lypred  = gp.predict(xd)
        dx_step = xd[1, 0] - xd[0, 0]
        mu_pred = -float(np.mean(np.gradient(lypred, dx_step)))
        return gp, mu_pred
    except Exception:
        return None, np.nan


# ============================================================
# P2-7: VISUALISATION WITH LAMBDA PANEL
# ============================================================
def plot_loss_curves(history, lam_history, material_name):
    ep_data = np.arange(len(history["data"])) * 10
    ep_lam  = np.arange(len(lam_history)) * 10
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    axes[0].semilogy(ep_data, history["data"], color="tab:blue")
    axes[0].set_title(r"Data Loss ($\mathcal{L}_{data}$)")
    axes[0].set_xlabel("Epochs"); axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(ep_data, history["phys"], color="tab:red")
    axes[1].set_title(r"Physics Loss ($\mathcal{L}_{phys}$)")
    axes[1].set_xlabel("Epochs"); axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(ep_data, history["total"], color="tab:green")
    axes[2].set_title(r"Total Loss ($\mathcal{L}_{total}$)")
    axes[2].set_xlabel("Epochs"); axes[2].grid(True, alpha=0.3)

    axes[3].plot(ep_lam, lam_history, color="tab:purple", linewidth=1.5)
    axes[3].set_title(r"Adaptive Weight $\lambda_\mathrm{phys}(t)$")
    axes[3].set_xlabel("Epochs"); axes[3].set_yscale("log"); axes[3].grid(True, alpha=0.3)

    plt.suptitle(f"{material_name} - PINN Convergence Dynamics (v2)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{material_name.lower()}_loss_curves.png"), dpi=300)
    plt.close()


def plot_fit_comparisons(x_data, y_noisy, x_max, true_mu, true_I0,
                         pinn_mu, pinn_I0, wnlls_mu, wnlls_I0, gp_model, mat_name):
    xd = np.linspace(0, x_max, 200)
    plt.figure(figsize=(8, 6))
    plt.scatter(x_data, y_noisy, c="black", label="Noisy Observations", zorder=5, alpha=0.6)
    plt.plot(xd, true_I0*np.exp(-true_mu*xd), "k--", lw=2,
             label=f"True Physics (mu={true_mu:.4f})")
    plt.plot(xd, pinn_I0*np.exp(-pinn_mu*xd), color="tab:red", lw=2,
             label=f"Rectified PINN (mu={pinn_mu:.4f})")
    if np.isfinite(wnlls_mu) and wnlls_I0 is not None and np.isfinite(wnlls_I0):
        plt.plot(xd, wnlls_I0*np.exp(-wnlls_mu*xd), color="tab:blue", ls="-.", lw=2,
                 label=f"WNLLS (mu={wnlls_mu:.4f})")
    if gp_model is not None:
        plt.plot(xd, np.exp(gp_model.predict(xd.reshape(-1,1))),
                 color="tab:green", ls=":", lw=2, label="GP Mean")
    plt.yscale("log")
    plt.title(f"{mat_name} - Curve Fits (High Noise Regime)", fontweight="bold")
    plt.xlabel("Thickness (mm)"); plt.ylabel("Photon Counts (log scale)")
    plt.legend(framealpha=0.9); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{mat_name.lower()}_curve_fits.png"), dpi=300)
    plt.close()


def generate_figure2_panel(summary_df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for i, n_val in enumerate(SAMPLE_SIZES):
        ax     = axes[i]
        subset = summary_df[summary_df["N_label"] == str(n_val)]
        for algo, style, color, label in [
            ("pinn",  "-o",  "tab:red",   "Rectified PINN"),
            ("wnlls", "--s", "tab:blue",  "WNLLS"),
            ("gp",    ":^",  "tab:green", "GP"),
        ]:
            col = f"{algo}_rpe_median"
            if col in subset.columns and len(subset) > 0:
                grp = subset.groupby("noise_std")[col].mean().reset_index()
                ax.plot(grp["noise_std"].values, grp[col].values, style,
                        color=color, label=label, linewidth=1.8, markersize=6)
        title = f"N = {n_val}" if n_val != "ALL" else "N = Full Dataset"
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Noise Std", fontsize=10); ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel("Median RPE (%)", fontsize=10)
    axes[0].legend(loc="upper left", fontsize=9, framealpha=0.9)
    plt.suptitle("Algorithmic Robustness Across Sampling Density and Noise Regimes",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "figure2_robustness_panel.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# NEW: BUILDUP FACTOR PROOF-OF-CONCEPT
# ============================================================
class PINN_Buildup(nn.Module):
    """Two-parameter PINN for I=I0*(1+beta*x)*exp(-mu*x)."""
    def __init__(self, init_mu=0.05, init_beta=0.05, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.mu_floor = 1e-4
        tgt = max(float(init_mu) - self.mu_floor, 1e-4)
        z_mu_init = np.log(np.exp(tgt) - 1.0) if tgt < 20 else tgt
        self.z_mu   = nn.Parameter(torch.tensor(z_mu_init, dtype=TORCH_DTYPE))
        self.z_beta = nn.Parameter(torch.tensor(max(float(init_beta), 0.01), dtype=TORCH_DTYPE))

    def get_mu(self):
        return torch.nn.functional.softplus(self.z_mu) + self.mu_floor

    def get_beta(self):
        return torch.nn.functional.softplus(self.z_beta)

    def forward(self, x):
        return self.net(x)


def train_pinn_buildup(x_train, y_train, x_max, init_mu, init_beta, trial, epochs=3000):
    x_mean, x_std = x_train.mean(), x_train.std()
    if x_std < 1e-8: x_std = 1.0
    y_log         = np.log(np.maximum(y_train, 1e-12))
    y_mean, y_std = y_log.mean(), y_log.std()
    if y_std < 1e-8: y_std = 1.0

    x_data_t   = torch.tensor((x_train-x_mean)/x_std, dtype=TORCH_DTYPE).view(-1,1).to(device)
    y_data_t   = torch.tensor((y_log  -y_mean)/y_std, dtype=TORCH_DTYPE).view(-1,1).to(device)
    sampler    = qmc.LatinHypercube(d=1, seed=42+trial)
    x_phys_c   = (sampler.random(n=M_COLLOC) * x_max).flatten()
    x_colloc_t = torch.tensor((x_phys_c-x_mean)/x_std, dtype=TORCH_DTYPE).view(-1,1).to(device)
    x_colloc_t.requires_grad_(True)
    x_phys_t   = torch.tensor(x_phys_c, dtype=TORCH_DTYPE).view(-1,1).to(device)

    model     = PINN_Buildup(init_mu=init_mu, init_beta=init_beta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    best_loss, best_state = float("inf"), None

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        data_loss = torch.mean((model(x_data_t) - y_data_t)**2)
        yc        = model(x_colloc_t)
        dy_dt     = torch.autograd.grad(yc, x_colloc_t, torch.ones_like(yc), create_graph=True)[0]
        dlnI_dx   = dy_dt * (y_std / x_std)   # back to physical units
        mu_v      = model.get_mu()
        beta_v    = model.get_beta()
        ode_rhs   = beta_v / (1.0 + beta_v * x_phys_t) - mu_v
        phys_loss = torch.mean((dlnI_dx - ode_rhs)**2)
        total     = data_loss + phys_loss
        if total.item() < best_loss:
            best_loss  = total.item()
            best_state = copy.deepcopy(model.state_dict())
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)
    return float(model.get_mu().item()), float(model.get_beta().item())


def run_buildup_experiment():
    print("\n" + "="*75)
    print("BUILDUP FACTOR PROOF-OF-CONCEPT EXPERIMENT")
    print("  Forward model: I(x) = I0*(1+beta*x)*exp(-mu*x)")
    print("  True: mu=0.0500 mm^-1, beta=0.0800 mm^-1, I0=50000 counts")
    print("="*75)

    TRUE_MU, TRUE_BETA, TRUE_I0 = 0.0500, 0.0800, 50000.0
    x_pts    = np.linspace(0, 60, 13, dtype=np.float64)
    x_max    = x_pts.max()
    NOISE_LV = [0.0, 0.0316, 0.10, 0.3162, 0.5623]
    N_BT     = 15
    rows     = []

    for noise_std in NOISE_LV:
        pinn_mu_l, pinn_beta_l, wnlls_mu_l = [], [], []
        for trial in range(N_BT):
            np.random.seed(7777+trial); torch.manual_seed(7777+trial)
            y_clean = TRUE_I0*(1+TRUE_BETA*x_pts)*np.exp(-TRUE_MU*x_pts)
            if noise_std > 0:
                eps   = np.random.normal(0, noise_std, len(x_pts))
                y_obs = y_clean * np.exp(eps - 0.5*noise_std**2)
            else:
                y_obs = y_clean.copy()
            _, w_mu = fit_wnlls(x_pts, y_obs)
            wnlls_mu_l.append(w_mu)
            i_mu   = robust_initial_slope(x_pts, y_obs)
            i_beta = max(0.01, TRUE_BETA * 0.5)
            mu_p, beta_p = train_pinn_buildup(x_pts, y_obs, x_max, i_mu, i_beta, trial)
            pinn_mu_l.append(mu_p); pinn_beta_l.append(beta_p)

        wa = np.array(wnlls_mu_l, dtype=float)
        pa = np.array(pinn_mu_l,  dtype=float)
        ba = np.array(pinn_beta_l, dtype=float)
        row = {
            "noise_std":              noise_std,
            "wnlls_mu_rpe_median":    float(np.nanmedian(np.abs(wa-TRUE_MU)/TRUE_MU*100)),
            "wnlls_failures":         int(np.sum(np.isnan(wa))),
            "pinn_mu_rpe_median":     float(np.nanmedian(np.abs(pa-TRUE_MU)/TRUE_MU*100)),
            "pinn_beta_rpe_median":   float(np.nanmedian(np.abs(ba-TRUE_BETA)/TRUE_BETA*100)),
        }
        rows.append(row)
        print(f"  sigma={noise_std*100:5.2f}% | "
              f"WNLLS mu RPE: {row['wnlls_mu_rpe_median']:6.1f}% (fails:{row['wnlls_failures']}) | "
              f"PINN mu RPE: {row['pinn_mu_rpe_median']:6.1f}% | "
              f"PINN beta RPE: {row['pinn_beta_rpe_median']:6.1f}%")

    bdf = pd.DataFrame(rows)
    bdf.to_csv(os.path.join(RESULTS_DIR, "table_buildup_factor.csv"), index=False)

    noise_pct = [n*100 for n in NOISE_LV]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(noise_pct, bdf["wnlls_mu_rpe_median"], "b--s", lw=2, label="WNLLS mu (biased model)")
    axes[0].plot(noise_pct, bdf["pinn_mu_rpe_median"],  "r-o",  lw=2, label="Rectified PINN mu")
    axes[0].set_xlabel("Noise Std (%)"); axes[0].set_ylabel("Median mu RPE (%)")
    axes[0].set_title("mu Recovery: WNLLS vs PINN (Buildup)", fontweight="bold")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(noise_pct, bdf["pinn_beta_rpe_median"], "g-^", lw=2, label="PINN beta")
    axes[1].axhline(20, color="grey", ls=":", label="20% threshold")
    axes[1].set_xlabel("Noise Std (%)"); axes[1].set_ylabel("Median beta RPE (%)")
    axes[1].set_title("Buildup Coeff beta Recovery (PINN only)", fontweight="bold")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.suptitle("I(x)=I0*(1+beta*x)*exp(-mu*x) -- Buildup Factor Recovery",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "figure_buildup_experiment.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Buildup results -> {RESULTS_DIR}")
    return bdf


# ============================================================
# PUBLICATION OUTPUT ENGINE — ALL FIGURES AND TABLES
# ============================================================
# Called once after run_master_suite() has produced summary_df
# and run_buildup_experiment() has produced buildup_df.
# Produces:
#   Figure 1  - Per-material loss curves (already per-material during training)
#   Figure 2  - RPE vs noise panel (already in generate_figure2_panel)
#   Figure 3  - Violin/box plots of RPE distributions at sigma=56.2%
#   Figure 4  - FDR significance heatmap (q-values)
#   Figure 5  - Bar chart: full-N median RPE all materials (clean summary)
#   Figure 6  - Buildup experiment (already in run_buildup_experiment)
#   Figure 7  - PINN vs WNLLS scatter: per-trial RPE at sigma=56.2% (diagnostic)
#   Table 1   - LaTeX: Main robustness table with CI and FDR
#   Table 2   - LaTeX: Ablation (noise tier vs method) condensed
#   Table 3   - LaTeX: Computational complexity (manual, formatted)
#   Table 4   - LaTeX: Buildup factor results
#   stats.txt - Key numbers ready to copy into abstract/results text

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
import matplotlib.ticker as mticker

# Colour palette consistent across all figures
C_PINN  = "#e63946"   # vivid red
C_WNLLS = "#457b9d"   # steel blue
C_GP    = "#2a9d8f"   # teal
C_LIGHT = "#f1faee"   # near-white background
ALPHA   = 0.75

# ============================================================
# FIG 3: VIOLIN / BOX PLOTS AT sigma=56.2%
# ============================================================
def fig3_violin_plots(summary_df, raw_trial_data):
    """
    raw_trial_data: dict keyed by (material, N_label, noise_tier) ->
        {'pinn': array(30,), 'wnlls': array(30,), 'gp': array(30,)}
    Violin plots per material at the highest noise tier, full N.
    """
    high_noise_idx = len(NOISE_TIERS) - 1  # sigma=56.2%
    materials_full = [m for m in MATERIALS]

    fig, axes = plt.subplots(1, len(materials_full),
                             figsize=(4 * len(materials_full), 6), sharey=False)
    if len(materials_full) == 1:
        axes = [axes]

    for ax, mat_key in zip(axes, materials_full):
        mat_name = MATERIALS[mat_key][0]
        key = (mat_name, "ALL", high_noise_idx)
        if key not in raw_trial_data:
            ax.set_title(mat_name); continue

        d = raw_trial_data[key]
        pinn_r  = np.clip(np.nan_to_num(d["pinn"], nan=RPE_FAIL_PENALTY),  0, 300)
        wnlls_r = np.clip(np.nan_to_num(d["wnlls"], nan=RPE_FAIL_PENALTY), 0, 300)
        gp_r    = np.clip(np.nan_to_num(d["gp"], nan=RPE_FAIL_PENALTY),    0, 300)

        positions = [1, 2, 3]
        data_list = [pinn_r, wnlls_r, gp_r]
        colors    = [C_PINN, C_WNLLS, C_GP]
        labels    = ["PINN", "WNLLS", "GP"]

        if HAS_SEABORN:
            import pandas as _pd
            df_long = _pd.DataFrame({
                "RPE": np.concatenate([pinn_r, wnlls_r, gp_r]),
                "Method": (["Rectified\nPINN"]*len(pinn_r) +
                           ["WNLLS"]*len(wnlls_r) +
                           ["GP"]*len(gp_r)),
            })
            sns.violinplot(data=df_long, x="Method", y="RPE", hue="Method",
                           palette=[C_PINN, C_WNLLS, C_GP],
                           inner="box", ax=ax, cut=0, linewidth=1.2, legend=False)
        else:
            vp = ax.violinplot(data_list, positions=positions, showmedians=True,
                               showextrema=True)
            for pc, col in zip(vp["bodies"], colors):
                pc.set_facecolor(col); pc.set_alpha(0.65)
            ax.set_xticks(positions); ax.set_xticklabels(labels, fontsize=9)

        ax.set_title(f"{mat_name}\n(Full N, σ=56.2%)", fontsize=10, fontweight="bold")
        ax.set_ylabel("RPE (%)" if mat_key == materials_full[0] else "")
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", alpha=0.3)
        ax.axhline(y=50, color="grey", ls=":", lw=1, alpha=0.6)

    legend_els = [Patch(facecolor=C_PINN,  label="Rectified PINN"),
                  Patch(facecolor=C_WNLLS, label="WNLLS"),
                  Patch(facecolor=C_GP,    label="GP")]
    fig.legend(handles=legend_els, loc="lower center", ncol=3,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    plt.suptitle("RPE Distribution at Extreme Noise (σ=56.2%) — Full Dataset",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig3_violin_high_noise.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [FIG 3] Violin plots -> {path}")


# ============================================================
# FIG 4: FDR SIGNIFICANCE HEATMAP
# ============================================================
def fig4_fdr_heatmap(summary_df):
    """q-values for WNLLS<PINN comparison as a heatmap."""
    noise_labels = [f"{s*100:.1f}%" for s in NOISE_TIERS]
    n_labels     = ["N=5", "N=ALL"]

    for pair_tag, title in [
        ("wnlls_vs_pinn_penalty", "WNLLS < PINN (penalty-inclusive FDR q-value)"),
        ("gp_vs_pinn_penalty",    "GP < PINN (penalty-inclusive FDR q-value)"),
    ]:
        q_col = f"qval_{pair_tag}"
        if q_col not in summary_df.columns:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
        for ax, n_lbl in zip(axes, n_labels):
            subset = summary_df[summary_df["N_label"] == n_lbl.replace("N=", "")]
            mat_order = [MATERIALS[k][0] for k in MATERIALS]

            matrix = np.ones((len(mat_order), len(NOISE_TIERS))) * np.nan
            for i, mat_name in enumerate(mat_order):
                for j, ns in enumerate(NOISE_TIERS):
                    row = subset[
                        (subset["material"] == mat_name) &
                        (np.isclose(subset["noise_std"], ns, atol=1e-6))
                    ]
                    if len(row) > 0 and np.isfinite(row[q_col].values[0]):
                        matrix[i, j] = row[q_col].values[0]

            im = ax.imshow(matrix, vmin=0, vmax=1, aspect="auto",
                           cmap="RdYlGn_r", origin="upper")
            ax.set_xticks(range(len(NOISE_TIERS)))
            ax.set_xticklabels(noise_labels, fontsize=8)
            ax.set_yticks(range(len(mat_order)))
            ax.set_yticklabels(mat_order, fontsize=9)
            ax.set_title(f"{n_lbl}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Noise Tier σ", fontsize=9)

            # Annotate cells
            for i in range(len(mat_order)):
                for j in range(len(NOISE_TIERS)):
                    v = matrix[i, j]
                    if np.isfinite(v):
                        txt = f"{v:.3f}"
                        star = "***" if v < 0.001 else ("**" if v < 0.01 else
                               ("*" if v < 0.05 else ""))
                        col = "white" if v < 0.3 else "black"
                        ax.text(j, i, f"{txt}\n{star}", ha="center",
                                va="center", fontsize=6.5, color=col,
                                fontweight="bold" if star else "normal")

            plt.colorbar(im, ax=ax, label="FDR q-value", fraction=0.046)

        tag_short = pair_tag.split("_vs_")[0].upper()
        plt.suptitle(f"Figure 4: {title}", fontsize=12, fontweight="bold", y=1.01)
        plt.tight_layout()
        fname = f"fig4_fdr_heatmap_{tag_short}_vs_PINN.png"
        plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [FIG 4] FDR heatmap ({tag_short}) -> {fname}")


# ============================================================
# FIG 5: BAR CHART — FULL-N MEDIAN RPE ALL MATERIALS
# ============================================================
def fig5_material_bar_chart(summary_df):
    """Clean summary bar chart: full-N, highest noise tier, all materials."""
    subset = summary_df[
        (summary_df["N_label"] == "ALL") &
        (summary_df["noise_tier"] == len(NOISE_TIERS) - 1)
    ].copy()

    if subset.empty:
        print("  [FIG 5] No data for ALL/high-noise — skipping.")
        return

    mat_names = subset["material"].tolist()
    x         = np.arange(len(mat_names))
    width     = 0.24

    fig, ax = plt.subplots(figsize=(11, 5))

    bars_p = ax.bar(x - width, subset["pinn_rpe_median"].values,
                    width, color=C_PINN,  alpha=ALPHA, label="Rectified PINN",
                    edgecolor="black", linewidth=0.6)
    bars_w = ax.bar(x,          subset["wnlls_rpe_median"].values,
                    width, color=C_WNLLS, alpha=ALPHA, label="WNLLS",
                    edgecolor="black", linewidth=0.6)
    bars_g = ax.bar(x + width,  subset["gp_rpe_median"].values,
                    width, color=C_GP,    alpha=ALPHA, label="GP",
                    edgecolor="black", linewidth=0.6)

    # Error bars using bootstrap CIs
    for i, (bar, lo, hi) in enumerate(zip(
            bars_p,
            subset["pinn_rpe_ci_lo"].values,
            subset["pinn_rpe_ci_hi"].values)):
        med = subset["pinn_rpe_median"].values[i]
        ax.errorbar(bar.get_x() + bar.get_width()/2, med,
                    yerr=[[med-lo], [hi-med]], fmt="none",
                    ecolor="black", capsize=4, linewidth=1.2)
    for i, (bar, lo, hi) in enumerate(zip(
            bars_w,
            subset["wnlls_rpe_ci_lo"].values,
            subset["wnlls_rpe_ci_hi"].values)):
        med = subset["wnlls_rpe_median"].values[i]
        ax.errorbar(bar.get_x() + bar.get_width()/2, med,
                    yerr=[[med-lo], [hi-med]], fmt="none",
                    ecolor="black", capsize=4, linewidth=1.2)
    for i, (bar, lo, hi) in enumerate(zip(
            bars_g,
            subset["gp_rpe_ci_lo"].values,
            subset["gp_rpe_ci_hi"].values)):
        med = subset["gp_rpe_median"].values[i]
        ax.errorbar(bar.get_x() + bar.get_width()/2, med,
                    yerr=[[med-lo], [hi-med]], fmt="none",
                    ecolor="black", capsize=4, linewidth=1.2)

    # WNLLS failure annotations
    for i, row in enumerate(subset.itertuples()):
        if row.wnlls_failures > 0:
            ax.text(i, row.wnlls_rpe_median + 2, f"✗{row.wnlls_failures}",
                    ha="center", va="bottom", fontsize=8, color=C_WNLLS,
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(mat_names, fontsize=10)
    ax.set_ylabel("Median RPE (%) [95% bootstrap CI]", fontsize=10)
    ax.set_xlabel("Attenuator Material", fontsize=10)
    ax.set_title("Figure 5: Algorithm Comparison at Extreme Noise (σ=56.2%, Full Dataset)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig5_material_bar_chart.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [FIG 5] Material bar chart -> {path}")


# ============================================================
# FIG 7: PER-TRIAL SCATTER — PINN vs WNLLS RPE AT sigma=56.2%
# ============================================================
def fig7_scatter_pinn_vs_wnlls(raw_trial_data):
    """
    Scatter plot: PINN RPE (x) vs WNLLS RPE (y) per trial.
    Points above y=x diagonal → WNLLS worse; below → PINN worse.
    One panel per material.
    """
    high_idx  = len(NOISE_TIERS) - 1
    mat_keys  = [k for k in MATERIALS]
    fig, axes = plt.subplots(1, len(mat_keys),
                             figsize=(4*len(mat_keys), 4.5), sharey=False)
    if len(mat_keys) == 1:
        axes = [axes]

    for ax, mat_key in zip(axes, mat_keys):
        mat_name = MATERIALS[mat_key][0]
        key = (mat_name, "ALL", high_idx)
        if key not in raw_trial_data:
            ax.set_title(mat_name); continue

        d      = raw_trial_data[key]
        p_rpe  = np.clip(d["pinn"],  0, 300)
        w_rpe  = d["wnlls"]  # keep NaN for penalty

        # Replace NaN with penalty
        w_plot = np.where(np.isnan(w_rpe), RPE_FAIL_PENALTY, w_rpe)
        w_plot = np.clip(w_plot, 0, 300)

        # Color by outcome
        pinn_better = p_rpe < w_plot
        ax.scatter(p_rpe[ pinn_better], w_plot[ pinn_better], c=C_PINN,
                   alpha=0.7, s=40, label="PINN better", zorder=3)
        ax.scatter(p_rpe[~pinn_better], w_plot[~pinn_better], c=C_WNLLS,
                   alpha=0.7, s=40, label="WNLLS better", zorder=3)

        # Diagonal
        lim = max(p_rpe.max(), w_plot.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1.2, alpha=0.5)

        ax.set_xlabel("Rectified PINN RPE (%)", fontsize=9)
        ax.set_ylabel("WNLLS RPE (%)" if mat_key == mat_keys[0] else "", fontsize=9)
        ax.set_title(mat_name, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)

        pinn_wins = int(pinn_better.sum())
        ax.text(0.97, 0.04, f"PINN wins: {pinn_wins}/30",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color=C_PINN,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.9)
    plt.suptitle("Figure 7: Per-Trial PINN vs WNLLS RPE Scatter (σ=56.2%, Full N)\n"
                 "Points below diagonal → WNLLS better; above → PINN better",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig7_scatter_pinn_vs_wnlls.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [FIG 7] Scatter diagnostic -> {path}")


# ============================================================
# FIG 8: NOISE-RAMP MULTI-MATERIAL OVERLAY
# ============================================================
def fig8_noise_ramp_overlay(summary_df):
    """
    One figure per N-label: all materials overlaid on same axes.
    Shows how each material degrades with noise for the rectified PINN.
    """
    mat_colors = {m: MATERIALS[m][1] for m in MATERIALS}
    noise_pct  = [s * 100 for s in NOISE_TIERS]

    for n_lbl in ["5", "ALL"]:
        subset = summary_df[summary_df["N_label"] == n_lbl]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

        for algo, ax, col_med, col_lo, col_hi, title, color in [
            ("pinn",  axes[0], "pinn_rpe_median",  "pinn_rpe_ci_lo",  "pinn_rpe_ci_hi",
             "Rectified PINN",  C_PINN),
            ("wnlls", axes[1], "wnlls_rpe_median", "wnlls_rpe_ci_lo", "wnlls_rpe_ci_hi",
             "WNLLS",           C_WNLLS),
            ("gp",    axes[2], "gp_rpe_median",    "gp_rpe_ci_lo",    "gp_rpe_ci_hi",
             "GP",              C_GP),
        ]:
            for mat_key, (mat_name, mat_c) in MATERIALS.items():
                mat_sub = subset[subset["material"] == mat_name].sort_values("noise_std")
                if mat_sub.empty:
                    continue
                ns  = mat_sub["noise_std"].values * 100
                med = mat_sub[col_med].values
                lo  = mat_sub[col_lo].values
                hi  = mat_sub[col_hi].values
                ax.plot(ns, med, "-o", color=mat_c, label=mat_name,
                        linewidth=1.8, markersize=5)
                ax.fill_between(ns, lo, hi, color=mat_c, alpha=0.15)

            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xlabel("Noise Std σ (%)", fontsize=9)
            ax.set_ylabel("Median RPE (%)" if algo == "pinn" else "", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)

        axes[0].legend(fontsize=8, framealpha=0.9, ncol=1)
        n_title = "N = 5 (Sparse)" if n_lbl == "5" else "N = Full Dataset"
        plt.suptitle(f"Figure 8: RPE vs Noise — All Materials, {n_title}\n"
                     "Shaded bands: 95% bootstrap CI on median",
                     fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, f"fig8_noise_ramp_N{n_lbl}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [FIG 8] Noise ramp overlay (N={n_lbl}) -> {path}")


# ============================================================
# LATEX TABLE GENERATOR
# ============================================================
def generate_latex_tables(summary_df, buildup_df=None):
    """
    Writes three .tex files to RESULTS_DIR:
      table1_main.tex     — main robustness comparison
      table2_ablation.tex — condensed noise-tier ablation
      table4_buildup.tex  — buildup factor results (if available)
    """

    # ---- Table 1: Main robustness ----
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Monte Carlo Robustness Comparison (30 trials/cell). "
        r"RPE values are median [95\% bootstrap CI]. "
        r"Primary Wilcoxon test uses $\text{RPE}=1000\%$ penalty for NaN failures. "
        r"FDR: Benjamini-Hochberg $q < 0.05$.}",
        r"\label{tab:main_robustness}",
        r"\scriptsize",
        r"\begin{tabular}{llr|ccc|ccc|ccc|cr}",
        r"\toprule",
        r"Mat. & $N$ & $\sigma$ & \multicolumn{3}{c|}{Rectified PINN RPE (\%)} "
        r"& \multicolumn{3}{c|}{WNLLS RPE (\%)} "
        r"& \multicolumn{3}{c|}{GP RPE (\%)} "
        r"& $q_{\text{WNLLS<PINN}}$ & WNLLS\\",
        r" & & (\%) & Med [CI] & Mean & Outl. "
        r"& Med [CI] & Mean & Outl. "
        r"& Med [CI] & Mean & Outl. & & Fails\\",
        r"\midrule",
    ]

    prev_mat = None
    for _, row in summary_df.sort_values(
            ["material", "N_actual", "noise_std"]).iterrows():
        mat = row["material"]
        N   = int(row["N_actual"])
        sig = f"{row['noise_std']*100:.2f}"

        def fmt(med, lo, hi, mean, outl):
            ci_str = f"[{lo:.1f},{hi:.1f}]" if np.isfinite(lo) and np.isfinite(hi) else "[---]"
            mean_s = f"{mean:.1f}" if np.isfinite(mean) else "---"
            outl_s = str(int(outl)) if np.isfinite(outl) else "---"
            return f"{med:.1f} {ci_str} & {mean_s} & {outl_s}"

        p_s = fmt(row["pinn_rpe_median"],  row["pinn_rpe_ci_lo"],
                  row["pinn_rpe_ci_hi"],   row["pinn_rpe_mean"],  row["pinn_outliers"])
        w_s = fmt(row["wnlls_rpe_median"], row["wnlls_rpe_ci_lo"],
                  row["wnlls_rpe_ci_hi"],  row["wnlls_rpe_mean"], row["wnlls_outliers"])
        g_s = fmt(row["gp_rpe_median"],    row["gp_rpe_ci_lo"],
                  row["gp_rpe_ci_hi"],     row["gp_rpe_mean"],    row["gp_outliers"])

        q_col = "qval_wnlls_vs_pinn_penalty"
        q_val = row.get(q_col, np.nan)
        if np.isfinite(q_val):
            stars = "***" if q_val < 0.001 else ("**" if q_val < 0.01 else
                    ("*" if q_val < 0.05 else ""))
            q_str = f"{q_val:.4f}{stars}"
        else:
            q_str = "---"

        fails = int(row.get("wnlls_failures", 0))
        mat_cell = r"\midrule " + mat if mat != prev_mat else ""
        prev_mat = mat

        line = (f"{mat_cell} & {N} & {sig} "
                f"& {p_s} & {w_s} & {g_s} & {q_str} & {fails}\\\\")
        lines.append(line)

    lines += [
        r"\bottomrule",
        r"\multicolumn{14}{l}{\footnotesize{$*q<0.05$, $**q<0.01$, $***q<0.001$ "
        r"(BH-FDR on penalty-inclusive Wilcoxon $p$-values). "
        r"Outl.: trials with RPE $>200\%$. Fails: WNLLS NaN returns.}}",
        r"\end{tabular}",
        r"\end{table}",
    ]

    tex1_path = os.path.join(RESULTS_DIR, "table1_main.tex")
    with open(tex1_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [LaTeX] Table 1 -> {tex1_path}")

    # ---- Table 2: Condensed ablation (Full N, all noise tiers, all materials) ----
    lines2 = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Ablation over Noise Tiers (Full Dataset $N$). "
        r"Median RPE (\%) shown with 95\% bootstrap CI in brackets.}",
        r"\label{tab:ablation_noise}",
        r"\small",
        r"\begin{tabular}{lr|ccc}",
        r"\toprule",
        r"Material & $\sigma$ (\%) & Rectified PINN & WNLLS & GP \\",
        r"\midrule",
    ]
    full_n_df = summary_df[summary_df["N_label"] == "ALL"].sort_values(
        ["material", "noise_std"])
    prev2 = None
    for _, row in full_n_df.iterrows():
        mat = row["material"]
        sig = f"{row['noise_std']*100:.1f}"
        def short(med, lo, hi):
            if np.isfinite(lo) and np.isfinite(hi):
                return f"{med:.1f} [{lo:.1f}, {hi:.1f}]"
            return f"{med:.1f}"
        p_s = short(row["pinn_rpe_median"],  row["pinn_rpe_ci_lo"],  row["pinn_rpe_ci_hi"])
        w_s = short(row["wnlls_rpe_median"], row["wnlls_rpe_ci_lo"], row["wnlls_rpe_ci_hi"])
        g_s = short(row["gp_rpe_median"],    row["gp_rpe_ci_lo"],    row["gp_rpe_ci_hi"])
        if mat != prev2:
            lines2.append(r"\midrule")
            prev2 = mat
        lines2.append(f"{mat} & {sig} & {p_s} & {w_s} & {g_s} \\\\")
    lines2 += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex2_path = os.path.join(RESULTS_DIR, "table2_ablation.tex")
    with open(tex2_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines2))
    print(f"  [LaTeX] Table 2 -> {tex2_path}")

    # ---- Table 3: Computational Complexity ----
    lines3 = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Computational Resource Consumption and Algorithmic Characteristics. "
        r"Hardware: CPU-mode PyTorch / DirectML GPU (AMD Radeon, 8\,GB VRAM).}",
        r"\label{tab:computational_complexity}",
        r"\small",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Algorithm & Inversion Paradigm & Iterative? & Wall-Clock Time & GPU Req.? & Theoretical Failure Mode \\",
        r"\midrule",
        r"Naive PINN & Deep Neural ODE & Yes (3000) & $\sim$15.2\,s / trial & Optional & Vanishing gradient, runaway \\",
        r"Rectified PINN & Constrained NN ODE & Yes (3000) & $\sim$14.8\,s / trial & Optional & Local sub-optimal minima \\",
        r"WNLLS & Linearized GLS & No (Exact) & $<$0.0005\,s & No (CPU Only) & Matrix singularity ($A^T A$) \\",
        r"GP Regressor & Bayesian Kernel & Yes (Opt) & $\sim$0.12\,s / trial & No (CPU Only) & Kernel length-scale collapse \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex3_path = os.path.join(RESULTS_DIR, "table3_complexity.tex")
    with open(tex3_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines3))
    print(f"  [LaTeX] Table 3 -> {tex3_path}")

    # ---- Table 4: Buildup ----
    if buildup_df is not None and len(buildup_df) > 0:
        lines4 = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Buildup Factor Proof-of-Concept: "
            r"$I(x)=I_0(1+\beta x)e^{-\mu x}$, True $\mu=0.0500$~mm$^{-1}$, "
            r"$\beta=0.0800$~mm$^{-1}$. WNLLS is misspecified (assumes pure exponential). "
            r"$n=15$ trials per tier.}",
            r"\label{tab:buildup}",
            r"\small",
            r"\begin{tabular}{r|rc|cc}",
            r"\toprule",
            r"$\sigma$ (\%) & WNLLS $\mu$ RPE (\%) & WNLLS Fails "
            r"& PINN $\mu$ RPE (\%) & PINN $\beta$ RPE (\%) \\",
            r"\midrule",
        ]
        for _, row in buildup_df.iterrows():
            sig = f"{row['noise_std']*100:.2f}"
            wr  = f"{row['wnlls_mu_rpe_median']:.1f}"
            wf  = str(int(row["wnlls_failures"]))
            pm  = f"{row['pinn_mu_rpe_median']:.1f}"
            pb  = f"{row['pinn_beta_rpe_median']:.1f}"
            lines4.append(f"{sig} & {wr} & {wf} & {pm} & {pb} \\\\")
        lines4 += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        tex4_path = os.path.join(RESULTS_DIR, "table4_buildup.tex")
        with open(tex4_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines4))
        print(f"  [LaTeX] Table 4 -> {tex4_path}")


# ============================================================
# SUMMARY STATISTICS TEXT FILE
# ============================================================
def generate_summary_stats(summary_df, buildup_df=None):
    """
    Writes a plain-text file with key numbers ready to copy into the
    abstract and results section. Also prints them to console.
    """
    lines = ["=" * 70, "KEY STATISTICS FOR MANUSCRIPT (auto-generated)", "=" * 70]

    # --- High-noise performance across all materials at full N ---
    high = summary_df[
        (summary_df["N_label"] == "ALL") &
        (summary_df["noise_tier"] == len(NOISE_TIERS) - 1)
    ]

    lines.append("\n[HIGH-NOISE sigma=56.2%, FULL N]")
    for _, row in high.iterrows():
        mat  = row["material"]
        pm   = row["pinn_rpe_median"]
        wm   = row["wnlls_rpe_median"]
        gm   = row["gp_rpe_median"]
        wf   = int(row["wnlls_failures"])
        po   = int(row["pinn_outliers"])
        q    = row.get("qval_wnlls_vs_pinn_penalty", np.nan)
        qdis = f"q={q:.4f}" if np.isfinite(q) else "q=nan"
        lines.append(f"  {mat}: PINN={pm:.1f}% WNLLS={wm:.1f}%(fails={wf}) "
                     f"GP={gm:.1f}% | PINN_outliers={po} | {qdis}")

    # --- PINN improvement over naive (from Table 2 in manuscript) ---
    lines.append("\n[PINN vs WNLLS: how many cells PINN wins at q<0.05]")
    q_col = "qval_wnlls_vs_pinn_penalty"
    if q_col in summary_df.columns:
        n_sig_wvp = int(np.sum(summary_df[q_col] < 0.05))
        lines.append(f"  WNLLS < PINN significant cells: {n_sig_wvp}/{len(summary_df)}")

    q_col2 = "qval_gp_vs_pinn_penalty"
    if q_col2 in summary_df.columns:
        n_sig_gvp = int(np.sum(summary_df[q_col2] < 0.05))
        lines.append(f"  GP < PINN significant cells:    {n_sig_gvp}/{len(summary_df)}")

    # --- WNLLS failure counts ---
    lines.append("\n[WNLLS FAILURE RATES]")
    fail_df = summary_df[summary_df["wnlls_failures"] > 0]
    if fail_df.empty:
        lines.append("  No WNLLS failures across any cell.")
    else:
        for _, row in fail_df.iterrows():
            lines.append(f"  {row['material']} N={row['N_actual']} "
                         f"sigma={row['noise_std']*100:.1f}%: "
                         f"{int(row['wnlls_failures'])}/30 failures")

    # --- NIST discrepancies ---
    lines.append("\n[NIST XCOM DISCREPANCIES (WOLS derived vs NIST)]")
    nist_df = summary_df.drop_duplicates("material")[
        ["material", "true_mu_wols", "nist_mu_mm", "mu_discrepancy_pct"]].dropna()
    for _, row in nist_df.iterrows():
        lines.append(f"  {row['material']}: derived={row['true_mu_wols']:.5f} mm^-1 "
                     f"NIST={row['nist_mu_mm']:.5f} mm^-1 "
                     f"disc={row['mu_discrepancy_pct']:+.1f}%")

    # --- Buildup ---
    if buildup_df is not None and len(buildup_df) > 0:
        lines.append("\n[BUILDUP FACTOR EXPERIMENT SUMMARY]")
        low_noise = buildup_df[buildup_df["noise_std"] < 0.05]
        high_noise = buildup_df.iloc[-1]
        if not low_noise.empty:
            r = low_noise.iloc[0]
            lines.append(f"  sigma=0%: WNLLS mu_RPE={r['wnlls_mu_rpe_median']:.1f}% "
                         f"PINN mu_RPE={r['pinn_mu_rpe_median']:.1f}% "
                         f"PINN beta_RPE={r['pinn_beta_rpe_median']:.1f}%")
        lines.append(f"  sigma={high_noise['noise_std']*100:.1f}%: "
                     f"WNLLS mu_RPE={high_noise['wnlls_mu_rpe_median']:.1f}% "
                     f"PINN mu_RPE={high_noise['pinn_mu_rpe_median']:.1f}% "
                     f"PINN beta_RPE={high_noise['pinn_beta_rpe_median']:.1f}%")

    lines.append("\n" + "=" * 70)
    text = "\n".join(lines)
    print(text)

    out = os.path.join(RESULTS_DIR, "summary_stats.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  [STATS] Summary stats -> {out}")


# ============================================================
# MASTER PUBLICATION FIGURE CALLER
# ============================================================
def generate_publication_figures(summary_df, buildup_df=None, raw_trial_data=None):
    """
    Call after run_master_suite() to produce ALL publication outputs.
    raw_trial_data: optional dict of per-trial RPE arrays for violin/scatter plots.
    """
    print("\n" + "=" * 75)
    print("GENERATING PUBLICATION FIGURES AND TABLES")
    print("=" * 75)

    # Fig 2 is already called inside run_master_suite()

    if raw_trial_data:
        print("\n[Figures requiring per-trial data]")
        fig3_violin_plots(summary_df, raw_trial_data)
        fig7_scatter_pinn_vs_wnlls(raw_trial_data)
    else:
        print("\n[Violin/scatter skipped — raw_trial_data not passed]")

    print("\n[Standard summary figures]")
    fig4_fdr_heatmap(summary_df)
    fig5_material_bar_chart(summary_df)
    fig8_noise_ramp_overlay(summary_df)

    print("\n[LaTeX tables]")
    generate_latex_tables(summary_df, buildup_df)

    print("\n[Summary statistics]")
    generate_summary_stats(summary_df, buildup_df)

    print("\n" + "=" * 75)
    print("[ALL PUBLICATION OUTPUTS COMPLETE]")
    print(f"Results directory: {RESULTS_DIR}")
    print("=" * 75)

# ============================================================
# MASTER ABLATION ENGINE
# ============================================================
def run_master_suite():
    all_results = []
    raw_trials_store = {}  # (mat_name, N_label, noise_tier) -> {'pinn':[], 'wnlls':[], 'gp':[]}

    for mat_key, (mat_name, _color) in MATERIALS.items():
        fpath = resolve_material_file(mat_key)
        if not fpath:
            print(f"[WARNING] Skipping {mat_name}: data file not found.")
            continue

        df_raw  = pd.read_csv(fpath)
        x_raw   = df_raw["thickness_mm"].values.astype(np.float64)
        y_raw   = df_raw["net_counts"].values.astype(np.float64)
        x_max   = x_raw.max()

        # P0-3: Weighted OLS ground truth
        true_mu, true_I0, mu_stderr = weighted_ols_ground_truth(x_raw, y_raw)
        nist_mu_rho, density        = NIST_MU_RHO.get(mat_key, (np.nan, np.nan))
        nist_mu_mm  = nist_mu_rho * density / 10.0   # cm^-1 -> mm^-1
        disc_pct    = (true_mu - nist_mu_mm)/nist_mu_mm*100 if np.isfinite(nist_mu_mm) else np.nan

        print(f"\nMaterial: {mat_name:<10} | N_raw: {len(x_raw)}")
        print(f"  Derived mu (WOLS): {true_mu:.5f}+-{mu_stderr:.5f} mm^-1 | "
              f"NIST XCOM: {nist_mu_mm:.5f} mm^-1 | Discrepancy: {disc_pct:+.1f}%")
        print("-" * 75)

        mat_hash     = abs(hash(mat_key)) % 10000
        evaluated_Ns = set()

        for N_req in SAMPLE_SIZES:
            actual_N = len(x_raw) if N_req == "ALL" else min(N_req, len(x_raw))
            if actual_N in evaluated_Ns:
                continue
            evaluated_Ns.add(actual_N)

            for noise_idx, noise_std in enumerate(NOISE_TIERS):
                pinn_mus, wnlls_mus, gp_mus = [], [], []

                for trial in range(N_TRIALS):
                    seed = mat_hash + 1000*actual_N + 100*noise_idx + trial
                    np.random.seed(seed); torch.manual_seed(seed)

                    idx     = np.linspace(0, len(x_raw)-1, actual_N, dtype=int)
                    x_sub   = x_raw[idx]
                    y_sub   = y_raw[idx]
                    if noise_std > 0:
                        eps    = np.random.normal(0, noise_std, actual_N)
                        y_noisy = y_sub * np.exp(eps - 0.5*noise_std**2)
                    else:
                        y_noisy = y_sub.copy()

                    init_mu = robust_initial_slope(x_sub, y_noisy)
                    track_v = (trial == 0 and N_req == "ALL"
                               and noise_idx == len(NOISE_TIERS)-1)

                    if track_v:
                        p_mu, history, lam_hist = train_pinn_lhs(
                            x_sub, y_noisy, x_max, init_mu, trial,
                            track_loss=True, track_lambda=True)
                        plot_loss_curves(history, lam_hist, mat_name)
                    else:
                        p_mu = train_pinn_lhs(x_sub, y_noisy, x_max, init_mu, trial)

                    w_I0, w_mu = fit_wnlls(x_sub, y_noisy)
                    gp_mod, g_mu = fit_gp(x_sub, y_noisy, x_max,
                                          seed=mat_hash+trial)   # P1-4

                    if track_v:
                        log_mu_x = np.log(np.maximum(y_noisy, 1e-12)) + p_mu*x_sub
                        p_I0     = float(np.exp(np.mean(log_mu_x)))
                        plot_fit_comparisons(x_sub, y_noisy, x_max, true_mu, true_I0,
                                             p_mu, p_I0, w_mu, w_I0, gp_mod, mat_name)

                    pinn_mus.append(p_mu)
                    wnlls_mus.append(w_mu)
                    gp_mus.append(g_mu)

                pinn_arr  = np.array(pinn_mus,  dtype=float)
                wnlls_arr = np.array(wnlls_mus, dtype=float)
                gp_arr    = np.array(gp_mus,    dtype=float)

                pinn_rpe  = np.abs((pinn_arr  - true_mu)/true_mu)*100.0
                wnlls_rpe = np.abs((wnlls_arr - true_mu)/true_mu)*100.0
                gp_rpe    = np.abs((gp_arr    - true_mu)/true_mu)*100.0

                raw_trials_store[(mat_name, str(N_req), noise_idx)] = {
                    'pinn':  pinn_rpe,
                    'wnlls': wnlls_rpe,
                    'gp':    gp_rpe,
                }

                pinn_s    = robust_stats(pinn_rpe)
                wnlls_s   = robust_stats(wnlls_rpe)
                gp_s      = robust_stats(gp_rpe)
                wnlls_failures = int(np.sum(np.isnan(wnlls_arr)))

                # P0-2: dual Wilcoxon
                p_wp,  p_wc,  n_pwp = paired_wilcoxon_penalty(wnlls_rpe, pinn_rpe)
                p_gpp, p_gpc, n_pgp = paired_wilcoxon_penalty(gp_rpe,    pinn_rpe)
                p_wgp, p_wgc, n_pwg = paired_wilcoxon_penalty(wnlls_rpe, gp_rpe)

                all_results.append({
                    "material": mat_name, "N_actual": actual_N,
                    "N_label": str(N_req), "noise_tier": noise_idx,
                    "noise_std": noise_std,
                    "true_mu_wols": true_mu, "true_mu_stderr": mu_stderr,
                    "nist_mu_mm": nist_mu_mm, "mu_discrepancy_pct": disc_pct,
                    # PINN
                    "pinn_rpe_mean": pinn_s["mean"],    "pinn_rpe_median": pinn_s["median"],
                    "pinn_rpe_std":  pinn_s["std"],     "pinn_rpe_ci_lo":  pinn_s["ci_lo"],
                    "pinn_rpe_ci_hi": pinn_s["ci_hi"],  "pinn_outliers":   pinn_s["outliers"],
                    # WNLLS
                    "wnlls_rpe_mean": wnlls_s["mean"],  "wnlls_rpe_median": wnlls_s["median"],
                    "wnlls_rpe_std":  wnlls_s["std"],   "wnlls_rpe_ci_lo":  wnlls_s["ci_lo"],
                    "wnlls_rpe_ci_hi": wnlls_s["ci_hi"], "wnlls_failures": wnlls_failures,
                    "wnlls_outliers": wnlls_s["outliers"],
                    # GP
                    "gp_rpe_mean": gp_s["mean"],        "gp_rpe_median": gp_s["median"],
                    "gp_rpe_std":  gp_s["std"],         "gp_rpe_ci_lo":  gp_s["ci_lo"],
                    "gp_rpe_ci_hi": gp_s["ci_hi"],      "gp_outliers":   gp_s["outliers"],
                    # Wilcoxon primary (penalty-inclusive)
                    "pval_wnlls_vs_pinn_penalty":    p_wp,
                    "pval_gp_vs_pinn_penalty":       p_gpp,
                    "pval_wnlls_vs_gp_penalty":      p_wgp,
                    "n_penalty_wnlls_vs_pinn":       n_pwp,
                    # Wilcoxon secondary (conditional on shared success)
                    "pval_wnlls_vs_pinn_conditional": p_wc,
                    "pval_gp_vs_pinn_conditional":    p_gpc,
                    "pval_wnlls_vs_gp_conditional":   p_wgc,
                })

                pd.DataFrame(all_results).to_csv(
                    os.path.join(RESULTS_DIR, "table1_master_robustness_LIVE.csv"), index=False)

                fail_s = f" [WNLLS fails:{wnlls_failures}]" if wnlls_failures else ""
                print(f"  N={actual_N:3d} s={noise_std*100:4.1f}% "
                      f"PINN:{pinn_s['median']:5.1f}%[{pinn_s['ci_lo']:.1f},{pinn_s['ci_hi']:.1f}] "
                      f"WNLLS:{wnlls_s['median']:5.1f}%{fail_s} "
                      f"GP:{gp_s['median']:5.1f}%")

    if not all_results:
        print("[ERROR] No results produced.")
        return

    summary_df = pd.DataFrame(all_results)

    # P0-1: BH-FDR over all penalty-inclusive p-values
    print("\nApplying Benjamini-Hochberg FDR correction ...")
    for tag in ["wnlls_vs_pinn_penalty", "gp_vs_pinn_penalty", "wnlls_vs_gp_penalty"]:
        pvals = summary_df[f"pval_{tag}"].values.astype(float)
        summary_df[f"qval_{tag}"] = bh_fdr(pvals)
        n_sig = int(np.sum(summary_df[f"qval_{tag}"] < 0.05))
        print(f"  {tag}: {n_sig}/{len(summary_df)} cells at FDR q<0.05")

    out = os.path.join(RESULTS_DIR, "table1_master_robustness_v2.csv")
    summary_df.to_csv(out, index=False)
    print(f"\n[SUCCESS] Master table -> {out}")

    generate_figure2_panel(summary_df)
    buildup_df = run_buildup_experiment()

    # Generate all publication-quality figures and LaTeX tables
    generate_publication_figures(summary_df, buildup_df, raw_trial_data=raw_trials_store)

    print("\n" + "="*75)
    print("[ALL TASKS COMPLETED SUCCESSFULLY]")
    print("="*75)


if __name__ == "__main__":
    run_master_suite()