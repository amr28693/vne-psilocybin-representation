#!/usr/bin/env python3
"""
══════════════════════════════════════
VON NEUMANN ENTROPY AND NETWORK DISSOLUTION UNDER PSILOCYBIN
──────────────────────────────────────
Data:   Madsen et al. (2021) Eur Neuropsychopharmacol 50:121-132
        Table S3: Network RSFC as linear function of plasma psilocin level
Method: VN entropy of 7×7 network-level FC matrices reconstructed across
        the psilocin pharmacokinetic trajectory
──────────────────
Code by A.M. Rodriguez (2026)
══════════════════

The Madsen data provides:
    FC(PPL) = Intercept + β × PPL

where Intercept is baseline (pre-drug) FC and β is the change per unit
plasma psilocin level (ng/mL). This allows reconstruction of the full
7×7 network FC matrix at any point along the pharmacokinetic trajectory.

Networks (Yeo-like):
    AN  = Auditory
    DMN = Default Mode
    DAN = Dorsal Attention
    ECN = Executive Control (≈ Frontoparietal)
    SAN = Salience (≈ Ventral Attention)
    SMN = Sensorimotor
    VN  = Visual

Usage:
    python psychedelic_tefi_madsen.py
    python psychedelic_tefi_madsen.py --ppl-max 20
    python psychedelic_tefi_madsen.py --ppl-max 25 --steps 200
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import linalg
import argparse
import os
import json
from datetime import datetime

# ═════════════════
# CONFIGURATION
# ═════════════════

NETWORKS = ["AN", "DMN", "DAN", "ECN", "SAN", "SMN", "VN"]
N_NETS = len(NETWORKS)

# Network display colors (consistent with neuroimaging conventions)
NET_COLORS = {
    "AN":  "#E69F00",  # amber
    "DMN": "#CC3311",  # red
    "DAN": "#009E73",  # green
    "ECN": "#0072B2",  # blue
    "SAN": "#D55E00",  # vermillion
    "SMN": "#56B4E9",  # sky blue
    "VN":  "#AA4499",  # purple
}

# Full network names for display
NET_NAMES = {
    "AN":  "Auditory",
    "DMN": "Default Mode",
    "DAN": "Dorsal Attention",
    "ECN": "Executive Control",
    "SAN": "Salience",
    "SMN": "Sensorimotor",
    "VN":  "Visual",
}

OUTPUT_DIR = "madsen_tefi_output"

# ═══════════════════
# MADSEN 2021 TABLE S3 DATA
# ═══════════════════
# Embedded directly — no external file dependency for reproducibility.
# Source: Table S3, Madsen et al. (2021) Eur Neuropsychopharmacol 50:121-132
# FC(PPL) = Intercept + β × PPL
# Intercept = baseline (pre-drug) FC
# β = slope of FC vs plasma psilocin level (ng/mL)

MADSEN_S3 = {
    # Within-network FC (diagonal)
    "AN":      {"intercept": 0.80,    "beta": -0.0028},
    "DMN":     {"intercept": 0.25,    "beta": -0.0036},
    "DAN":     {"intercept": 0.44,    "beta": -0.0037},
    "ECN":     {"intercept": 0.42,    "beta": -0.0058},
    "SAN":     {"intercept": 0.42,    "beta": -0.0067},   
    "SMN":     {"intercept": 0.45,    "beta":  0.0026},
    "VN":      {"intercept": 1.40,    "beta":  0.00019},
    # Between-network FC (off-diagonal, symmetric)
    "DMN-AN":  {"intercept": -0.09,   "beta": -0.0012},
    "DMN-DAN": {"intercept": -0.15,   "beta":  0.0052},   
    "DMN-ECN": {"intercept":  0.0022, "beta":  0.0065},   
    "DMN-SAN": {"intercept": -0.097,  "beta":  0.0048},   
    "DMN-SMN": {"intercept": -0.042,  "beta":  0.0018},
    "DMN-VN":  {"intercept":  0.0062, "beta": -0.0011},
    "DAN-AN":  {"intercept":  0.081,  "beta":  0.0089},   
    "DAN-ECN": {"intercept":  0.079,  "beta": -0.0022},
    "DAN-SAN": {"intercept":  0.049,  "beta":  0.000023},
    "DAN-SMN": {"intercept":  0.093,  "beta":  0.011},    
    "DAN-VN":  {"intercept":  0.017,  "beta":  0.0037},
    "ECN-AN":  {"intercept": -0.078,  "beta":  0.0023},
    "ECN-SAN": {"intercept":  0.10,   "beta": -0.00056},
    "ECN-SMN": {"intercept": -0.10,   "beta":  0.0055}, 
    "ECN-VN":  {"intercept": -0.047,  "beta": -0.00036},
    "SAN-AN":  {"intercept":  0.21,   "beta":  0.0014},
    "SAN-SMN": {"intercept":  0.018,  "beta":  0.0012},
    "SAN-VN":  {"intercept": -0.085,  "beta":  0.004},
    "SMN-AN":  {"intercept":  0.17,   "beta":  0.0016},
    "SMN-VN":  {"intercept":  0.056,  "beta": -0.0056},
    "VN-AN":   {"intercept":  0.021,  "beta":  0.0000870},
}

# Statistical significance from original paper (pFWER < 0.05)
SIGNIFICANT_WITHIN = {"DMN", "SAN"}  # Within-network FC significantly ↓ with PPL
SIGNIFICANT_BETWEEN = {"DMN-ECN", "DMN-SAN", "DAN-AN", "DAN-SMN"}  # Between significantly ↑


# ══════════════════════
# CORE FUNCTIONS
# ══════════════════════

def build_fc_matrix(ppl):
    """
    Reconstruct 7×7 network-level FC matrix at a given plasma psilocin level.

    Parameters
    ----------
    ppl : float
        Plasma psilocin level in ng/mL (0 = baseline/pre-drug)

    Returns
    -------
    fc : ndarray (7, 7)
        Symmetric FC matrix with within-network on diagonal
    """
    fc = np.zeros((N_NETS, N_NETS))

    # Diagonal: within-network FC
    for i, net in enumerate(NETWORKS):
        d = MADSEN_S3[net]
        fc[i, i] = d["intercept"] + d["beta"] * ppl

    # Off-diagonal: between-network FC
    for key, d in MADSEN_S3.items():
        if "-" not in key:
            continue
        parts = key.split("-")
        i = NETWORKS.index(parts[0])
        j = NETWORKS.index(parts[1])
        val = d["intercept"] + d["beta"] * ppl
        fc[i, j] = val
        fc[j, i] = val

    return fc


def von_neumann_entropy(R, epsilon=1e-12):
    """
    Von Neumann entropy of a correlation/FC matrix.

    S(ρ) = -tr(ρ log ρ) where ρ = R/tr(R)

    Parameters
    ----------
    R : ndarray (n, n)
        Symmetric FC matrix (may have negative off-diagonal)
    epsilon : float
        Floor for near-zero eigenvalues

    Returns
    -------
    S : float
        VN entropy in nats
    S_norm : float
        S / S_max where S_max = log(n)
    eigenvalues : ndarray
        Eigenvalues of the density matrix ρ
    """
    # Ensure symmetry
    R = (R + R.T) / 2

    # Eigendecompose
    eigvals = linalg.eigvalsh(R)

    # Handle negative eigenvalues (can arise from negative FC values)
    # Shift spectrum to make all eigenvalues non-negative
    if eigvals.min() < 0:
        eigvals = eigvals - eigvals.min() + epsilon

    # Normalize to density matrix (trace = 1)
    total = eigvals.sum()
    if total < epsilon:
        return 0.0, 0.0, eigvals
    rho_eigvals = eigvals / total

    # VN entropy
    rho_eigvals = rho_eigvals[rho_eigvals > epsilon]
    S = -np.sum(rho_eigvals * np.log(rho_eigvals))
    S_max = np.log(len(eigvals))
    S_norm = S / S_max if S_max > 0 else 0.0

    return S, S_norm, rho_eigvals


def network_dissolution_index(fc):
    """
    Dissolution index: ratio of mean |between-network FC| to mean within-network FC.

    Higher values indicate more dissolved network boundaries.
    At perfect segregation, off-diagonal → 0, index → 0.
    """
    diag = np.abs(np.diag(fc))
    mask = ~np.eye(N_NETS, dtype=bool)
    off_diag = np.abs(fc[mask])
    return off_diag.mean() / diag.mean() if diag.mean() > 0 else np.inf


def compute_tefi_analog(fc):
    """
    TEFI-analog for network-level matrix.

    In Golino et al. (2020), TEFI = sum of within-community VN entropies
    evaluated against hypothesized community structure. At the network level,
    perfect community structure = block-diagonal matrix (off-diagonal = 0).

    We compute: the VN entropy of the matrix with off-diagonal elements
    included vs. excluded. The difference quantifies how much the between-
    network coupling contributes to the total entropy — i.e., how much
    network boundaries have dissolved.

    TEFI_analog = S(full matrix) - S(diagonal-only matrix)

    Higher TEFI_analog = more dissolution beyond what within-network
    structure alone would predict.
    """
    S_full, _, _ = von_neumann_entropy(fc)

    # Diagonal-only matrix (perfect segregation)
    fc_diag = np.diag(np.diag(fc))
    S_diag, _, _ = von_neumann_entropy(fc_diag)

    return S_full - S_diag


# ═══════════════════════
# ANALYSIS
# ═══════════════════════

def run_trajectory(ppl_max=20.0, n_steps=100):
    """
    Compute VN entropy trajectory across the psilocin pharmacokinetic curve.

    Madsen et al. report PPL peaking at ~15-20 ng/mL around 80-130 min
    post-administration, with typical range 0-20 ng/mL.
    """
    ppl_values = np.linspace(0, ppl_max, n_steps)

    results = {
        "ppl": ppl_values,
        "S_vn": np.zeros(n_steps),
        "S_norm": np.zeros(n_steps),
        "dissolution_index": np.zeros(n_steps),
        "tefi_analog": np.zeros(n_steps),
        "mean_within_fc": np.zeros(n_steps),
        "mean_between_fc": np.zeros(n_steps),
    }

    # Per-network within-FC trajectories
    for net in NETWORKS:
        results[f"within_{net}"] = np.zeros(n_steps)

    # Per-network between-FC trajectories (mean of all between-connections)
    for net in NETWORKS:
        results[f"between_{net}"] = np.zeros(n_steps)

    for idx, ppl in enumerate(ppl_values):
        fc = build_fc_matrix(ppl)

        # Global VN entropy
        S, S_norm, eigvals = von_neumann_entropy(fc)
        results["S_vn"][idx] = S
        results["S_norm"][idx] = S_norm

        # Dissolution index
        results["dissolution_index"][idx] = network_dissolution_index(fc)

        # TEFI analog
        results["tefi_analog"][idx] = compute_tefi_analog(fc)

        # Within-network FC (diagonal)
        diag = np.diag(fc)
        results["mean_within_fc"][idx] = diag.mean()
        for i, net in enumerate(NETWORKS):
            results[f"within_{net}"][idx] = diag[i]

        # Between-network FC per network (mean of row excluding diagonal)
        mask = ~np.eye(N_NETS, dtype=bool)
        results["mean_between_fc"][idx] = fc[mask].mean()
        for i, net in enumerate(NETWORKS):
            between_vals = [fc[i, j] for j in range(N_NETS) if j != i]
            results[f"between_{net}"][idx] = np.mean(between_vals)

    return results


# ═══════════════════════
# FIGURES
# ═══════════════════════

def fig1_entropy_trajectory(results, output_dir):
    """
    Fig 1: VN entropy and dissolution metrics vs. plasma psilocin level.
    Three-panel figure: (A) S/Smax, (B) TEFI analog, (C) Dissolution index
    """
    ppl = results["ppl"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # (A) Normalized VN entropy
    ax = axes[0]
    ax.plot(ppl, results["S_norm"], "k-", linewidth=2)
    ax.set_xlabel("Plasma psilocin level (ng/mL)")
    ax.set_ylabel("S / S$_{max}$")
    ax.set_title("(A) Von Neumann entropy", fontsize=11, fontweight="bold")
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)

    # Mark baseline
    ax.annotate("Baseline", xy=(0, results["S_norm"][0]),
                xytext=(2, results["S_norm"][0] - 0.005),
                fontsize=8, color="gray")

    # (B) TEFI analog
    ax = axes[1]
    ax.plot(ppl, results["tefi_analog"], "k-", linewidth=2)
    ax.set_xlabel("Plasma psilocin level (ng/mL)")
    ax.set_ylabel("TEFI analog (nats)")
    ax.set_title("(B) Between-network entropy contribution", fontsize=11, fontweight="bold")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)

    # (C) Dissolution index
    ax = axes[2]
    ax.plot(ppl, results["dissolution_index"], "k-", linewidth=2)
    ax.set_xlabel("Plasma psilocin level (ng/mL)")
    ax.set_ylabel("Dissolution index\n(|between| / within FC)")
    ax.set_title("(C) Network boundary dissolution", fontsize=11, fontweight="bold")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(f"{output_dir}/figures/fig1_entropy_trajectory.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close()
    print("  fig1_entropy_trajectory.png/pdf")


def fig2_network_integrity(results, output_dir):
    """
    Fig 2: Per-network within-FC trajectories.
    Shows which networks lose integrity fastest under psilocin.
    """
    ppl = results["ppl"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (A) Within-network FC trajectories
    ax = axes[0]
    for net in NETWORKS:
        baseline = results[f"within_{net}"][0]
        trajectory = results[f"within_{net}"]
        # Plot as % change from baseline
        pct_change = (trajectory - baseline) / np.abs(baseline) * 100
        style = "-" if net in SIGNIFICANT_WITHIN else "--"
        lw = 2.5 if net in SIGNIFICANT_WITHIN else 1.2
        alpha = 1.0 if net in SIGNIFICANT_WITHIN else 0.6
        ax.plot(ppl, pct_change, style, color=NET_COLORS[net],
                linewidth=lw, alpha=alpha, label=f"{net} ({NET_NAMES[net]})")

    ax.set_xlabel("Plasma psilocin level (ng/mL)")
    ax.set_ylabel("Within-network FC (% change from baseline)")
    ax.set_title("(A) Network integrity", fontsize=11, fontweight="bold")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower left", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (B) Bar chart: β values (slope of within-FC vs PPL)
    ax = axes[1]
    betas = [MADSEN_S3[net]["beta"] for net in NETWORKS]
    colors = [NET_COLORS[net] for net in NETWORKS]
    edgecolors = ["black" if net in SIGNIFICANT_WITHIN else "gray" for net in NETWORKS]
    linewidths = [2 if net in SIGNIFICANT_WITHIN else 0.5 for net in NETWORKS]

    bars = ax.bar(range(N_NETS), betas, color=colors, edgecolor=edgecolors,
                  linewidth=linewidths)
    ax.set_xticks(range(N_NETS))
    ax.set_xticklabels(NETWORKS, fontsize=9)
    ax.set_ylabel("β (FC change per ng/mL psilocin)")
    ax.set_title("(B) Within-network dissolution rate", fontsize=11, fontweight="bold")
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate significant networks
    for i, net in enumerate(NETWORKS):
        if net in SIGNIFICANT_WITHIN:
            ax.annotate("*", xy=(i, betas[i]),
                        xytext=(i, betas[i] - 0.0008),
                        fontsize=14, ha="center", fontweight="bold")

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(f"{output_dir}/figures/fig2_network_integrity.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close()
    print("  fig2_network_integrity.png/pdf")


def fig3_desegregation(results, output_dir):
    """
    Fig 3: Between-network FC changes — which network boundaries dissolve.
    """
    ppl = results["ppl"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (A) Per-network mean between-FC trajectory
    ax = axes[0]
    for net in NETWORKS:
        baseline = results[f"between_{net}"][0]
        trajectory = results[f"between_{net}"]
        delta = trajectory - baseline
        ax.plot(ppl, delta, "-", color=NET_COLORS[net],
                linewidth=1.8, label=f"{net}")

    ax.set_xlabel("Plasma psilocin level (ng/mL)")
    ax.set_ylabel("Δ Mean between-network FC")
    ax.set_title("(A) Network desegregation by network", fontsize=11, fontweight="bold")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
    ax.legend(fontsize=8, ncol=2, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (B) Heatmap: between-network β values
    ax = axes[1]
    beta_matrix = np.zeros((N_NETS, N_NETS))
    for key, d in MADSEN_S3.items():
        if "-" not in key:
            continue
        parts = key.split("-")
        i = NETWORKS.index(parts[0])
        j = NETWORKS.index(parts[1])
        beta_matrix[i, j] = d["beta"]
        beta_matrix[j, i] = d["beta"]

    # Diagonal = within-network betas
    for i, net in enumerate(NETWORKS):
        beta_matrix[i, i] = MADSEN_S3[net]["beta"]

    vmax = np.max(np.abs(beta_matrix)) * 0.9
    im = ax.imshow(beta_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="equal")
    ax.set_xticks(range(N_NETS))
    ax.set_yticks(range(N_NETS))
    ax.set_xticklabels(NETWORKS, fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(NETWORKS, fontsize=9)
    ax.set_title("(B) β-matrix: FC change rate", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, label="β (FC / ng·mL⁻¹)", shrink=0.8)

    # Mark significant entries
    for key in SIGNIFICANT_BETWEEN:
        parts = key.split("-")
        i = NETWORKS.index(parts[0])
        j = NETWORKS.index(parts[1])
        ax.plot(j, i, "k*", markersize=10)
        ax.plot(i, j, "k*", markersize=10)
    for net in SIGNIFICANT_WITHIN:
        i = NETWORKS.index(net)
        ax.plot(i, i, "k*", markersize=10)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(f"{output_dir}/figures/fig3_desegregation.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close()
    print("  fig3_desegregation.png/pdf")


def fig4_fc_matrices(output_dir, ppl_levels=[0, 5, 10, 15, 20]):
    """
    Fig 4: Side-by-side FC matrices at different psilocin levels.
    Visual comparison of network structure dissolution.
    """
    n_panels = len(ppl_levels)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 3.5))

    # Use consistent color scale across all panels
    all_matrices = [build_fc_matrix(p) for p in ppl_levels]
    vmax = max(np.max(np.abs(m)) for m in all_matrices) * 0.8

    for idx, (ppl, fc) in enumerate(zip(ppl_levels, all_matrices)):
        ax = axes[idx]
        im = ax.imshow(fc, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_xticks(range(N_NETS))
        ax.set_yticks(range(N_NETS))
        ax.set_xticklabels(NETWORKS, fontsize=7, rotation=45, ha="right")
        ax.set_yticklabels(NETWORKS, fontsize=7)

        label = "Baseline" if ppl == 0 else f"PPL = {ppl}"
        ax.set_title(label, fontsize=10, fontweight="bold")

    plt.colorbar(im, ax=axes[-1], label="FC", shrink=0.8)
    fig.suptitle("Network FC matrices across psilocin trajectory",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(f"{output_dir}/figures/fig4_fc_matrices.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close()
    print("  fig4_fc_matrices.png/pdf")


def fig5_eigenvalue_spectra(output_dir, ppl_levels=[0, 10, 20]):
    """
    Fig 5: Eigenvalue spectra of the density matrix at different PPL levels.
    Flattening spectrum = increasing VN entropy = dissolution.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    cmap = plt.cm.viridis
    colors = [cmap(x) for x in np.linspace(0, 0.85, len(ppl_levels))]

    for ppl, color in zip(ppl_levels, colors):
        fc = build_fc_matrix(ppl)
        _, _, rho_eigvals = von_neumann_entropy(fc)
        rho_sorted = np.sort(rho_eigvals)[::-1]  # descending
        label = "Baseline" if ppl == 0 else f"PPL = {ppl}"
        ax.plot(range(1, N_NETS + 1), rho_sorted, "o-", color=color,
                linewidth=2, markersize=8, label=label)

    # Reference: maximum entropy (uniform)
    ax.axhline(y=1/N_NETS, color="gray", linestyle="--", alpha=0.5,
               label=f"Uniform (1/{N_NETS})")

    ax.set_xlabel("Eigenvalue rank")
    ax.set_ylabel("ρ eigenvalue (normalized)")
    ax.set_title("Density matrix eigenvalue spectra", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xticks(range(1, N_NETS + 1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(f"{output_dir}/figures/fig5_eigenvalue_spectra.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close()
    print("  fig5_eigenvalue_spectra.png/pdf")


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════════════

def print_summary(results):
    """Print key findings."""

    ppl = results["ppl"]
    idx_0 = 0
    idx_peak = np.argmax(ppl >= 15)  # ~peak PPL
    idx_max = -1

    print("\n  BASELINE (PPL = 0)")
    print(f"  ──────────────────────────────────────")
    print(f"    S/Smax:            {results['S_norm'][idx_0]:.6f}")
    print(f"    TEFI analog:       {results['tefi_analog'][idx_0]:.6f}")
    print(f"    Dissolution index: {results['dissolution_index'][idx_0]:.4f}")
    print(f"    Mean within-FC:    {results['mean_within_fc'][idx_0]:.4f}")
    print(f"    Mean between-FC:   {results['mean_between_fc'][idx_0]:.4f}")

    print(f"\n  PEAK EFFECT (PPL ≈ {ppl[idx_peak]:.0f} ng/mL)")
    print(f"  ──────────────────────────────────────")
    print(f"    S/Smax:            {results['S_norm'][idx_peak]:.6f}")
    print(f"    TEFI analog:       {results['tefi_analog'][idx_peak]:.6f}")
    print(f"    Dissolution index: {results['dissolution_index'][idx_peak]:.4f}")
    print(f"    Mean within-FC:    {results['mean_within_fc'][idx_peak]:.4f}")
    print(f"    Mean between-FC:   {results['mean_between_fc'][idx_peak]:.4f}")

    print(f"\n  CHANGE (Δ: peak - baseline)")
    print(f"  ──────────────────────────────────────")
    dS = results['S_norm'][idx_peak] - results['S_norm'][idx_0]
    dT = results['tefi_analog'][idx_peak] - results['tefi_analog'][idx_0]
    dD = results['dissolution_index'][idx_peak] - results['dissolution_index'][idx_0]
    print(f"    ΔS/Smax:            {dS:+.6f}")
    print(f"    ΔTEFI analog:       {dT:+.6f}")
    print(f"    ΔDissolution index: {dD:+.4f}")

    print(f"\n  PER-NETWORK DISSOLUTION RATE (β, within-FC)")
    print(f"  ──────────────────────────────────────────────────")
    print(f"  {'Network':<22s}  {'β':>10s}  {'pFWER':>8s}  {'Direction':>10s}")
    print(f"  {'─'*60}")

    # Sort by beta (most negative first = most dissolution)
    sorted_nets = sorted(NETWORKS, key=lambda n: MADSEN_S3[n]["beta"])
    for net in sorted_nets:
        b = MADSEN_S3[net]["beta"]
        sig = "< 0.05*" if net in SIGNIFICANT_WITHIN else "> 0.05"
        direction = "↓ dissolving" if b < 0 else "↑ strengthening"
        print(f"  {NET_NAMES[net]:<22s}  {b:>+10.4f}  {sig:>8s}  {direction:>12s}")


def save_results_csv(results, output_dir):
    """Save trajectory data to CSV."""
    df = pd.DataFrame({
        "ppl_ng_mL": results["ppl"],
        "S_vn": results["S_vn"],
        "S_norm": results["S_norm"],
        "tefi_analog": results["tefi_analog"],
        "dissolution_index": results["dissolution_index"],
        "mean_within_fc": results["mean_within_fc"],
        "mean_between_fc": results["mean_between_fc"],
    })

    # Add per-network columns
    for net in NETWORKS:
        df[f"within_{net}"] = results[f"within_{net}"]
        df[f"between_{net}"] = results[f"between_{net}"]

    path = f"{output_dir}/results/entropy_trajectory.csv"
    df.to_csv(path, index=False, float_format="%.8f")
    print(f"  {path}")
    return df


def save_fc_matrices(output_dir, ppl_levels=[0, 5, 10, 15, 20]):
    """Save reconstructed FC matrices."""
    matrices = {}
    for ppl in ppl_levels:
        fc = build_fc_matrix(ppl)
        key = f"ppl_{ppl:.0f}"
        matrices[key] = fc.tolist()

    path = f"{output_dir}/results/fc_matrices.json"
    with open(path, "w") as f:
        json.dump({
            "networks": NETWORKS,
            "ppl_levels": ppl_levels,
            "matrices": matrices,
            "description": "7x7 network FC matrices reconstructed from Madsen et al. (2021) Table S3"
        }, f, indent=2)
    print(f"  {path}")


# ═════════════════════════
# MAIN
# ═════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VN entropy analysis of psychedelic network dissolution (Madsen 2021 data)")
    parser.add_argument("--ppl-max", type=float, default=20.0,
                        help="Maximum plasma psilocin level to model (default: 20 ng/mL)")
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of steps in PPL trajectory (default: 100)")
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("VON NEUMANN ENTROPY AND NETWORK DISSOLUTION UNDER PSILOCYBIN")
    print("Data: Madsen et al. (2021) Eur Neuropsychopharmacol 50:121-132")
    print("=" * 70)

    # Setup output directories
    os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/results", exist_ok=True)

    # ── Data verification ─────────────
    print(f"\n(1) DATA")
    print(f"{'─' * 40}")
    print(f"  Source: Table S3 (PPL vs network FC)")
    print(f"  Networks: {N_NETS} ({', '.join(NETWORKS)})")
    print(f"  Within-network entries: {N_NETS}")
    n_between = sum(1 for k in MADSEN_S3 if "-" in k)
    print(f"  Between-network entries: {n_between}")
    print(f"  Total: {N_NETS + n_between}")
    print(f"  PPL range: 0 – {args.ppl_max:.0f} ng/mL ({args.steps} steps)")

    # Verify the matrix is complete
    expected_between = N_NETS * (N_NETS - 1) // 2
    assert n_between == expected_between, \
        f"Expected {expected_between} between-network entries, got {n_between}"
    print(f"  ✓ Complete 7×7 matrix verified ({expected_between} pairs)")

    # ── Analysis ─────────────────
    print(f"\n(2) ANALYSIS")
    print(f"{'─' * 40}")
    results = run_trajectory(ppl_max=args.ppl_max, n_steps=args.steps)
    print("  ✓ Entropy trajectory computed")

    # ── Summary ────────────────
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print_summary(results)

    # ── Save results ───────────────
    print(f"\n{'=' * 70}")
    print("OUTPUTS")
    print(f"{'=' * 70}")
    print(f"\n  Results:")
    save_results_csv(results, OUTPUT_DIR)
    save_fc_matrices(OUTPUT_DIR)

    # ── Figures ─────────────────
    print(f"\n  Figures:")
    fig1_entropy_trajectory(results, OUTPUT_DIR)
    fig2_network_integrity(results, OUTPUT_DIR)
    fig3_desegregation(results, OUTPUT_DIR)
    fig4_fc_matrices(OUTPUT_DIR)
    fig5_eigenvalue_spectra(OUTPUT_DIR)

    # ── Final summary ──────────────
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Data: Madsen et al. (2021), n=15 healthy volunteers")
    print(f"  Drug: Psilocybin 0.2-0.3 mg/kg oral")
    print(f"  Metric: VN entropy of 7×7 network FC matrix")
    print(f"  Trajectory: {args.steps} points, PPL 0–{args.ppl_max:.0f} ng/mL")

    # Key finding
    idx_0 = 0
    idx_peak = np.argmax(results["ppl"] >= 15)
    dS = results["S_norm"][idx_peak] - results["S_norm"][idx_0]
    direction = "↑ INCREASES" if dS > 0 else "↓ DECREASES"
    print(f"\n  Key finding: VN entropy {direction} with psilocin")
    print(f"    ΔS/Smax = {dS:+.6f} (baseline → PPL=15)")

    # Network hierarchy
    sorted_nets = sorted(NETWORKS, key=lambda n: MADSEN_S3[n]["beta"])
    hierarchy = " > ".join(sorted_nets[:4])
    print(f"    Dissolution hierarchy: {hierarchy}")
    print(f"    (by within-network FC slope, most negative first)")

    print(f"\n  All outputs in: {OUTPUT_DIR}/")
    print()


if __name__ == "__main__":
    main()
