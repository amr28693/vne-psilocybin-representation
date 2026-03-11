# Von Neumann Entropy Explorer

**Interactive companion to Rodriguez & Cates (2026)**

Demonstrates the paper's central finding: VNE reports properties of the chosen matrix representation, not properties of the underlying neural reorganization.

## Quick Start

```bash
pip install numpy
python vne_tool.py
```

Requires Python 3.8+, NumPy, and Tkinter (included on macOS/Windows; Linux: `sudo apt install python3-tk`).

## The Entropy Paradox in 30 Seconds

1. **Launch.** Baseline loads. Note **S/Smax = 0.876**, DI = 0.127.

2. **Click "② Peak psilocin".** Entropy *drops* to **0.823** while dissolution index *rises* to 0.147. The eigenvalue chart shows λ₅ and λ₆ collapsing.

3. **Click "③ Pearson-style".** S/Smax jumps to ~0.96. This is the representation where Felippe et al. (2021) found VNE *increases* under ayahuasca—opposite sign, same drug class.

4. **Return to baseline. Edit VN (bottom-right diagonal)** from 1.400 → 0.250. Watch entropy *increase*. Removing trace anchoring resolves the paradox.

## What You're Seeing

| Panel | Shows |
|-------|-------|
| **The Entropy Paradox** (blue box) | Reference values: baseline vs peak, with Δ displayed |
| **Current Matrix** | Live S/Smax, VNE, trace, dissolution index |
| **Eigenvalue Spectrum** | Bar chart with uniform-distribution reference line |
| **Interpretation** (yellow box) | Context-sensitive explanation of current state |

The key insight: under psilocybin, association networks (SAN, ECN, DMN) dissolve while the visual network (VN = 1.40) is pharmacologically spared. This trace anchoring collapses minor eigenvalues, *decreasing* VNE despite genuine network dissolution.

## Citation

Rodriguez, A.M.; Cates, M. Von Neumann Entropy Decreases Under Psilocybin Despite Network Dissolution: Implications for Scalar Entropy Measures of the Psychedelic State. 2026.
