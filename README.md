# Physics-Informed Neural Inversion of Narrow-Beam Gamma-Ray Attenuation

Official repository and experimental replication suite for the manuscript:
**"Failure Modes and Rectified Physics-Informed Neural Networks in Narrow-Beam Gamma-Ray Attenuation: An Experimental and Computational Benchmark"**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## 📌 Overview

This benchmark investigates the performance boundaries of Physics-Informed Neural Networks (PINNs) versus classical Weighted Non-Linear Least Squares (WNLLS) and Gaussian Process (GP) regression in recovering gamma-ray linear attenuation coefficients ($\mu$) from experimental narrow-beam transmission measurements.

Key findings:
- In low-to-moderate counting noise ($\sigma \le 10\%$), classical closed-form WNLLS is computationally optimal ($>30{,}000\times$ faster than PINNs) and achieves lower error (BLUE property).
- Under severe observational noise ($\sigma = 56.23\%$), classical WNLLS suffers from near-singular normal equations $(\mathbf{A}^T \mathbf{W} \mathbf{A})$, yielding a $13.3\%$ numerical collapse rate ($N=5$).
- The proposed **Rectified PINN** (Softplus physical constraint $\mu = \ln(1+e^z) + \mu_{\text{floor}}$, Theil-Sen robust slope initialization, and unweighted physical loss metric checkpointing) completely eliminates vanishing gradient traps and non-physical runaway modes ($0.0\%$ failure rate across 1,500 independent trials).

---

## 🔬 Experimental Data

Transmission measurements were acquired using a dedicated narrow-beam geometry with a sealed **Cs-137** radioisotope disc ($661.7\text{ keV}$), a $5.0\text{ mm}$ cylindrical lead collimator ($\theta_{\text{acc}} \approx 2.9^\circ$), and a $51\text{ mm} \times 51\text{ mm}$ $\text{NaI(Tl)}$ scintillation detector coupled to an Ortec MCA (dwell time $t=300\text{ s}$, dead-time $<1.5\%$).

Five physical engineering attenuators are provided in `data/`:
1. **Aluminum** ($Z=13$, $N=13$ thickness slabs from $0$ to $60\text{ mm}$)
2. **Copper** ($Z=29$, $N=6$ slabs from $0$ to $25\text{ mm}$)
3. **Brass** ($\text{Cu-Zn}$ alloy, $N=6$ slabs from $0$ to $25\text{ mm}$)
4. **Structural Carbon Steel** ($\text{Fe-C}$, $N=15$ slabs from $0$ to $35\text{ mm}$)
5. **Lead** ($Z=82$, $N=16$ slabs from $0$ to $22.5\text{ mm}$)

---

## 🚀 Installation & Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/4rayaditya/PINN-for-gamma-rays.git
cd PINN-for-gamma-rays
pip install -r requirements.txt
```

### Dependencies
- Python 3.10+
- PyTorch >= 2.0
- NumPy, SciPy, Pandas
- Scikit-Learn
- Matplotlib, Seaborn, Pillow

---

## 📊 Running the Benchmark

To execute the complete 1,500-trial Monte Carlo evaluation grid across all 5 materials, 2 sample sizes ($N=5$ and full $N$), and 5 noise tiers ($\sigma \in \{0.0\%, 3.16\%, 10.0\%, 31.62\%, 56.23\%\}$), along with the broad-beam buildup factor experiment:

```bash
python final_v2.py
```

Outputs will be automatically saved to `results_v2/`:
- **LaTeX Tables**: `table1_main.tex`, `table2_ablation.tex`, `table3_complexity.tex`, `table4_buildup.tex`
- **Master Figures**:
  - `figure2_robustness_panel.png`: RPE scaling across noise tiers
  - `fig3_violin_high_noise.png`: Violin distributions per material at $\sigma=56.2\%$
  - `fig4_fdr_heatmap_*.png`: Benjamini-Hochberg FDR significance heatmaps
  - `fig5_material_bar_chart.png`: Overall median RPE bar charts
  - `fig7_scatter_pinn_vs_wnlls.png`: Trial-by-trial paired scatter comparisons
  - `fig8_noise_ramp_*.png`: Multi-material noise overlay curves
  - Transmission curve fits and loss convergence curves for each material

---

## 📜 Citation & License

This project is licensed under the MIT License - see the LICENSE file for details.
