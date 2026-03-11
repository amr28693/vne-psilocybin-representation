# vne-psilocybin-representation

**Von Neumann entropy of network-level functional connectivity decreases under psilocybin despite network dissolution—demonstrating that scalar entropy measures are representation-dependent.**

Companion code for:

> Rodriguez, A.M.; Cates, M. (2026). Von Neumann Entropy Decreases Under Psilocybin Despite Network Dissolution: Implications for Scalar Entropy Measures of the Psychedelic State.

## Key Finding

VNE decreases (ΔS/Smax = −0.052) across the psilocybin pharmacokinetic trajectory while network boundaries dissolve. The paradox arises from selective 5-HT₂A-mediated dissolution of association networks while the high-connectivity visual network is spared, causing trace contraction and spectral concentration rather than the flattening VNE would require to increase.

Critically, Felippe et al. (2021) found the *opposite* sign (VNE increases) under ayahuasca using Pearson correlation matrices with fixed unit diagonals, i.e., the same drug class, same entropy measure yielded opposite results which are determined entirely by representational choice.

## Contents

| File | Description |
|------|-------------|
| `psychedelic_tefi_madsen.py` | Main analysis script. Reconstructs 7×7 FC matrices from Madsen et al. (2021) Table S3, computes VNE trajectory, generates figures. |
| `vne_tool.py` | Interactive GUI for exploring VNE. Demonstrates the entropy paradox in ~30 seconds with preset comparisons and live eigenvalue visualization. |
| `VNE_Tool_Writeup.md` | Documentation for the interactive tool. |

## Quick Start

```bash
# Main analysis
pip install numpy scipy pandas matplotlib
python psychedelic_tefi_madsen.py

# Interactive tool
python vne_tool.py
```

Outputs are written to `madsen_tefi_output/`.

## Data Source

Madsen, M.K. et al. (2021). Psilocybin-induced changes in brain network integrity and segregation correlate with plasma psilocin level and psychedelic experience. *Eur Neuropsychopharmacol* 50:121–132.

Table S3 regression coefficients are embedded directly in the code—no external data files required.

## Results Summary

| Metric | Baseline (PPL=0) | Peak (PPL=15) | Δ |
|--------|------------------|---------------|---|
| S/Smax | 0.876 | 0.824 | −0.052 |
| Dissolution index | 0.040 | 0.060 | +0.020 |

Dissolution hierarchy (by within-network FC slope): SAN > ECN > DAN > DMN > AN

Visual and sensorimotor networks are pharmacologically spared.

## Requirements

- Python 3.8+
- NumPy, SciPy, Pandas, Matplotlib
- Tkinter (for `vne_tool.py`; included in standard Python distributions)

## License

MIT

## Citation

```bibtex
@article{rodriguez2026vne,
  title={Von Neumann Entropy Decreases Under Psilocybin Despite Network Dissolution: 
         Implications for Scalar Entropy Measures of the Psychedelic State},
  author={Rodriguez, Anderson M. and Cates, Matthew},
  year={2026}
}
```
