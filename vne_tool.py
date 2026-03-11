"""
Von Neumann Entropy Explorer
Rodriguez & Cates (2026)
VNE = -tr(rho log rho), rho = R/tr(R), spectral-shifted for positive-definiteness.
"""

import tkinter as tk
from tkinter import ttk
import numpy as np

# ── Presets ──────────────────────────────────────────────────────────────────
# Matrices reconstructed from Madsen et al. (2021) Table S3 regression coefficients
# FC(PPL) = intercept + beta * PPL
# Network order: AN, DMN, DAN, ECN, SAN, SMN, VN

NETWORKS = ["AN", "DMN", "DAN", "ECN", "SAN", "SMN", "VN"]
NETWORK_NAMES = {
    "AN": "Auditory", "DMN": "Default Mode", "DAN": "Dorsal Attention",
    "ECN": "Executive Control", "SAN": "Salience", "SMN": "Sensorimotor", "VN": "Visual"
}

# PPL = 0 (baseline, pre-drug)
MADSEN_BASELINE = np.array([
    [ 0.800, -0.090,  0.081, -0.078,  0.210,  0.170,  0.021],
    [-0.090,  0.250, -0.150,  0.0022,-0.097, -0.042,  0.0062],
    [ 0.081, -0.150,  0.440,  0.079,  0.049,  0.093,  0.017],
    [-0.078,  0.0022, 0.079,  0.420,  0.100, -0.100, -0.047],
    [ 0.210, -0.097,  0.049,  0.100,  0.420,  0.018, -0.085],
    [ 0.170, -0.042,  0.093, -0.100,  0.018,  0.450,  0.056],
    [ 0.021,  0.0062, 0.017, -0.047, -0.085,  0.056,  1.400],
])

# PPL = 15 ng/mL (approximate pharmacokinetic peak)
MADSEN_PEAK = np.array([
    [ 0.758, -0.108,  0.2145, -0.0435,  0.231,   0.194,   0.0223],
    [-0.108,  0.196, -0.072,   0.0997, -0.025,  -0.015,  -0.0103],
    [ 0.2145,-0.072,  0.3845,  0.046,   0.0493,  0.258,   0.0725],
    [-0.0435, 0.0997, 0.046,   0.333,   0.0916, -0.0175, -0.0524],
    [ 0.231, -0.025,  0.0493,  0.0916,  0.3195,  0.036,  -0.025],
    [ 0.194, -0.015,  0.258,  -0.0175,  0.036,   0.489,  -0.028],
    [ 0.0223,-0.0103, 0.0725, -0.0524, -0.025,  -0.028,   1.40285],
])

PRESET_INFO = {
    "madsen_base": {
        "name": "Madsen Baseline",
        "desc": "Pre-psilocybin resting state (PPL = 0 ng/mL). Networks intact.",
    },
    "madsen_peak": {
        "name": "Madsen Peak",
        "desc": "Peak psilocin (PPL = 15 ng/mL). Association networks dissolved; VN spared.",
    },
    "pearson": {
        "name": "Pearson-style",
        "desc": "Unit diagonal (invariant to drug). Off-diagonal encodes all change → VNE increases.",
    },
    "identity": {
        "name": "Identity",
        "desc": "Reference: uniform eigenvalues → maximum entropy.",
    },
    "blank": {
        "name": "Blank",
        "desc": "Zero matrix (invalid density matrix).",
    },
}

def pearson_style(n):
    m = np.full((n, n), 0.12)
    np.fill_diagonal(m, 1.0)
    return m

def identity(n):
    return np.eye(n)

def blank(n):
    return np.zeros((n, n))


# ── Core computation ──────────────────────────────────────────────────────────

def compute_vne(matrix):
    n = len(matrix)
    if n == 0:
        return None
    M = np.array(matrix, dtype=float)
    sym = (M + M.T) / 2
    if not np.all(np.isfinite(sym)):
        return None

    raw_eigs = np.linalg.eigvalsh(sym)
    trace = np.diag(sym).sum()
    eig_sum = raw_eigs.sum()
    if abs(eig_sum) < 1e-12:
        return None

    eps = 1e-12
    min_eig = raw_eigs.min()
    shifted = raw_eigs - min_eig + eps if min_eig < eps else raw_eigs.copy()
    shifted_trace = shifted.sum()
    lambdas = shifted / shifted_trace
    lambdas = np.sort(lambdas)[::-1]

    vne = -sum(l * np.log(l) for l in lambdas if l > 0)
    max_vne = np.log(n)
    normalized_vne = vne / max_vne

    diag_vals = np.diag(sym)
    off_vals = sym[~np.eye(n, dtype=bool)]
    di = (np.mean(np.abs(off_vals)) / np.mean(diag_vals)) if np.mean(diag_vals) > 0 else 0.0

    return {
        "vne": vne,
        "normalized_vne": normalized_vne,
        "max_vne": max_vne,
        "lambdas": lambdas.tolist(),
        "trace": float(trace),
        "di": float(di),
        "n": n,
    }


# ── Reference values ──────────────────────────────────────────────────────────

BASELINE_REF = compute_vne(MADSEN_BASELINE)
PEAK_REF = compute_vne(MADSEN_PEAK)


# ── GUI ───────────────────────────────────────────────────────────────────────

class VNEApp:
    # High-contrast colors
    BG = "#f5f5f5"
    FG = "#1a1a1a"
    ACCENT = "#2563eb"  # blue
    ACCENT_DARK = "#1d4ed8"
    DIAG_BG = "#d4d4d4"
    CELL_BG = "#ffffff"
    BTN_BG = "#e5e5e5"
    BTN_FG = "#1a1a1a"
    BTN_ACTIVE_BG = "#2563eb"
    BTN_ACTIVE_FG = "#ffffff"
    
    FONT = ("Helvetica", 11)
    FONT_SM = ("Helvetica", 10)
    FONT_MONO = ("Courier", 11)
    FONT_MONO_SM = ("Courier", 9)
    FONT_BOLD = ("Helvetica", 11, "bold")
    FONT_TITLE = ("Helvetica", 14, "bold")
    FONT_BIG = ("Helvetica", 18, "bold")

    def __init__(self, root):
        self.root = root
        self.root.title("Von Neumann Entropy Explorer — Rodriguez & Cates (2026)")
        self.root.configure(bg=self.BG)

        self.n = 7
        self.sym_lock = tk.BooleanVar(value=True)
        self.matrix = MADSEN_BASELINE.tolist()
        self.labels = list(NETWORKS)
        self.active_preset = tk.StringVar(value="madsen_base")
        self.cell_vars = []

        self._build_ui()
        self._populate_cells()
        self._recompute()

    def _build_ui(self):
        root = self.root

        # ── Header ────────────────────────────────────────────────────────
        hdr = tk.Frame(root, bg=self.BG)
        hdr.pack(fill="x", padx=20, pady=(15, 5))
        
        tk.Label(hdr, text="VON NEUMANN ENTROPY EXPLORER",
                 font=self.FONT_TITLE, bg=self.BG, fg=self.FG).pack(anchor="w")
        tk.Label(hdr, text="Demonstrates representation-dependence of scalar entropy measures",
                 font=self.FONT_SM, bg=self.BG, fg="#666").pack(anchor="w")

        # Separator
        tk.Frame(root, height=2, bg="#ccc").pack(fill="x", padx=20, pady=(8, 12))

        # ── Main body ─────────────────────────────────────────────────────
        body = tk.Frame(root, bg=self.BG)
        body.pack(fill="both", expand=True, padx=20)

        # Left: presets + matrix
        self.left = tk.Frame(body, bg=self.BG)
        self.left.pack(side="left", anchor="n", padx=(0, 20))

        # Right: readouts
        self.right = tk.Frame(body, bg=self.BG)
        self.right.pack(side="left", anchor="n", fill="both", expand=True)

        self._build_presets()
        self._build_matrix_grid()
        self._build_readout()

        # ── Footer ────────────────────────────────────────────────────────
        tk.Frame(root, height=1, bg="#ccc").pack(fill="x", padx=20, pady=(12, 0))
        ftr = tk.Frame(root, bg=self.BG)
        ftr.pack(fill="x", padx=20, pady=(6, 12))
        tk.Label(ftr, text="Data: Madsen et al. (2021) Eur Neuropsychopharmacol 50:121-132",
                 font=self.FONT_SM, bg=self.BG, fg="#888").pack(anchor="w")

    def _build_presets(self):
        f = tk.Frame(self.left, bg=self.BG)
        f.pack(anchor="w", pady=(0, 12))

        tk.Label(f, text="LOAD PRESET", font=self.FONT_BOLD, 
                 bg=self.BG, fg=self.FG).pack(anchor="w", pady=(0, 6))

        # Preset buttons in a grid for clarity
        btn_frame = tk.Frame(f, bg=self.BG)
        btn_frame.pack(anchor="w")

        self._preset_btns = []
        presets = [
            ("madsen_base", "① Baseline (pre-drug)"),
            ("madsen_peak", "② Peak psilocin"),
            ("pearson",     "③ Pearson-style"),
            ("identity",    "④ Identity matrix"),
        ]
        
        for i, (key, label) in enumerate(presets):
            btn = tk.Button(btn_frame, text=label, font=self.FONT_SM,
                            width=20, anchor="w", padx=8, pady=4,
                            bg=self.BTN_BG, fg=self.BTN_FG,
                            activebackground=self.ACCENT, activeforeground="#fff",
                            relief="solid", bd=1, cursor="hand2",
                            command=lambda k=key: self._apply_preset(k))
            btn.grid(row=i//2, column=i%2, padx=2, pady=2, sticky="ew")
            btn._preset_key = key
            self._preset_btns.append(btn)

        # Preset description
        self._preset_desc = tk.Label(f, text="", font=self.FONT_SM, 
                                      bg=self.BG, fg="#555", wraplength=380,
                                      justify="left")
        self._preset_desc.pack(anchor="w", pady=(8, 0))

        # Size selector and symmetry lock
        ctrl_frame = tk.Frame(f, bg=self.BG)
        ctrl_frame.pack(anchor="w", pady=(10, 0))
        
        tk.Label(ctrl_frame, text="Matrix size:", font=self.FONT_SM,
                 bg=self.BG, fg=self.FG).pack(side="left")
        
        self._size_btns = []
        for v in [3, 4, 5, 6, 7, 8, 9, 10]:
            btn = tk.Button(ctrl_frame, text=str(v), font=self.FONT_SM,
                            width=2, bg=self.BTN_BG, fg=self.BTN_FG,
                            relief="solid", bd=1, cursor="hand2",
                            command=lambda x=v: self._change_n(x))
            btn.pack(side="left", padx=1)
            self._size_btns.append(btn)

        tk.Checkbutton(ctrl_frame, text="Enforce symmetry", variable=self.sym_lock,
                       font=self.FONT_SM, bg=self.BG, fg=self.FG,
                       activebackground=self.BG, selectcolor="#fff").pack(side="left", padx=(12, 0))

        self._update_preset_buttons()
        self._update_size_buttons()

    def _build_matrix_grid(self):
        outer = tk.Frame(self.left, bg=self.BG)
        outer.pack(anchor="w", pady=(8, 0))
        
        tk.Label(outer, text="FC MATRIX", font=self.FONT_BOLD,
                 bg=self.BG, fg=self.FG).pack(anchor="w", pady=(0, 4))
        tk.Label(outer, text="Diagonal = within-network FC · Off-diagonal = between-network FC",
                 font=self.FONT_SM, bg=self.BG, fg="#666").pack(anchor="w", pady=(0, 6))

        self.grid_frame = tk.Frame(outer, bg=self.BG)
        self.grid_frame.pack(anchor="w")
        self._rebuild_grid()

    def _rebuild_grid(self):
        for w in self.grid_frame.winfo_children():
            w.destroy()
        self.cell_vars = []

        # Column headers
        tk.Label(self.grid_frame, text="", width=4, bg=self.BG).grid(row=0, column=0)
        for j in range(self.n):
            lbl = self.labels[j] if j < len(self.labels) else f"C{j+1}"
            tk.Label(self.grid_frame, text=lbl, font=self.FONT_MONO_SM,
                     bg=self.BG, fg="#555", width=6).grid(row=0, column=j+1)

        # Matrix cells
        for i in range(self.n):
            row_vars = []
            lbl = self.labels[i] if i < len(self.labels) else f"R{i+1}"
            tk.Label(self.grid_frame, text=lbl, font=self.FONT_MONO_SM,
                     bg=self.BG, fg="#555", width=4, anchor="e").grid(row=i+1, column=0, padx=(0, 4))

            for j in range(self.n):
                var = tk.StringVar()
                bg = self.DIAG_BG if i == j else self.CELL_BG
                entry = tk.Entry(self.grid_frame, textvariable=var, width=6,
                                 font=self.FONT_MONO_SM, bg=bg, fg=self.FG,
                                 justify="center", relief="solid", bd=1,
                                 insertbackground=self.FG)
                entry.grid(row=i+1, column=j+1, padx=1, pady=1)
                var.trace_add("write", lambda *_, i=i, j=j: self._cell_changed(i, j))
                row_vars.append(var)
            self.cell_vars.append(row_vars)

    def _build_readout(self):
        r = self.right

        # ── Key comparison box ────────────────────────────────────────────
        compare_box = tk.Frame(r, bg="#e0e7ff", relief="solid", bd=1)
        compare_box.pack(fill="x", pady=(0, 12))
        
        tk.Label(compare_box, text="THE ENTROPY PARADOX", font=self.FONT_BOLD,
                 bg="#e0e7ff", fg=self.FG).pack(anchor="w", padx=12, pady=(10, 4))
        
        paradox_text = (
            "Under psilocybin, networks dissolve (↑ dissolution index) but entropy DECREASES.\n"
            "The visual network's high baseline FC anchors the trace, collapsing minor eigenvalues."
        )
        tk.Label(compare_box, text=paradox_text, font=self.FONT_SM,
                 bg="#e0e7ff", fg="#333", justify="left").pack(anchor="w", padx=12, pady=(0, 8))

        ref_frame = tk.Frame(compare_box, bg="#e0e7ff")
        ref_frame.pack(fill="x", padx=12, pady=(0, 10))
        
        # Reference values
        tk.Label(ref_frame, text="Baseline:", font=self.FONT_SM,
                 bg="#e0e7ff", fg="#555").grid(row=0, column=0, sticky="w")
        tk.Label(ref_frame, text=f"S/Smax = {BASELINE_REF['normalized_vne']:.3f}", 
                 font=self.FONT_MONO, bg="#e0e7ff", fg=self.FG).grid(row=0, column=1, padx=(8, 20))
        tk.Label(ref_frame, text=f"DI = {BASELINE_REF['di']:.3f}",
                 font=self.FONT_MONO, bg="#e0e7ff", fg=self.FG).grid(row=0, column=2)

        tk.Label(ref_frame, text="Peak PPL:", font=self.FONT_SM,
                 bg="#e0e7ff", fg="#555").grid(row=1, column=0, sticky="w")
        tk.Label(ref_frame, text=f"S/Smax = {PEAK_REF['normalized_vne']:.3f}", 
                 font=self.FONT_MONO, bg="#e0e7ff", fg=self.FG).grid(row=1, column=1, padx=(8, 20))
        tk.Label(ref_frame, text=f"DI = {PEAK_REF['di']:.3f}",
                 font=self.FONT_MONO, bg="#e0e7ff", fg=self.FG).grid(row=1, column=2)

        delta_s = PEAK_REF['normalized_vne'] - BASELINE_REF['normalized_vne']
        delta_di = PEAK_REF['di'] - BASELINE_REF['di']
        tk.Label(ref_frame, text="Change:", font=self.FONT_SM,
                 bg="#e0e7ff", fg="#555").grid(row=2, column=0, sticky="w")
        tk.Label(ref_frame, text=f"ΔS/Smax = {delta_s:+.3f}", 
                 font=("Courier", 11, "bold"), bg="#e0e7ff", 
                 fg="#dc2626" if delta_s < 0 else "#16a34a").grid(row=2, column=1, padx=(8, 20))
        tk.Label(ref_frame, text=f"ΔDI = {delta_di:+.3f}",
                 font=("Courier", 11, "bold"), bg="#e0e7ff", 
                 fg="#16a34a" if delta_di > 0 else "#dc2626").grid(row=2, column=2)

        # ── Current matrix readout ────────────────────────────────────────
        curr_box = tk.Frame(r, bg="#fff", relief="solid", bd=1)
        curr_box.pack(fill="x", pady=(0, 12))

        tk.Label(curr_box, text="CURRENT MATRIX", font=self.FONT_BOLD,
                 bg="#fff", fg=self.FG).pack(anchor="w", padx=12, pady=(10, 6))

        self._readout_labels = {}
        fields = [
            ("S / Smax", "Normalized VNE (0 = concentrated, 1 = uniform)"),
            ("VNE (nats)", "Raw Von Neumann entropy"),
            ("Matrix trace", "Sum of diagonal (within-network FC)"),
            ("Dissolution index", "Mean |between| / mean within FC"),
        ]
        
        for field, tooltip in fields:
            row = tk.Frame(curr_box, bg="#fff")
            row.pack(fill="x", padx=12, pady=2)
            
            lbl = tk.Label(row, text=field, font=self.FONT_SM, bg="#fff", fg="#555",
                           width=18, anchor="w")
            lbl.pack(side="left")
            
            val_lbl = tk.Label(row, text="--", font=self.FONT_MONO,
                               bg="#fff", fg=self.FG)
            val_lbl.pack(side="left", padx=(8, 0))
            self._readout_labels[field] = val_lbl

        # Big S/Smax display
        big_frame = tk.Frame(curr_box, bg="#fff")
        big_frame.pack(fill="x", padx=12, pady=(8, 12))
        
        self._big_entropy_label = tk.Label(big_frame, text="S/Smax = --", 
                                            font=self.FONT_BIG, bg="#fff", fg=self.ACCENT)
        self._big_entropy_label.pack(side="left")
        
        self._delta_label = tk.Label(big_frame, text="", font=self.FONT_BOLD,
                                      bg="#fff", fg="#666")
        self._delta_label.pack(side="left", padx=(16, 0))

        # ── Eigenvalue spectrum ───────────────────────────────────────────
        eig_box = tk.Frame(r, bg="#fff", relief="solid", bd=1)
        eig_box.pack(fill="x", pady=(0, 12))

        tk.Label(eig_box, text="EIGENVALUE SPECTRUM", font=self.FONT_BOLD,
                 bg="#fff", fg=self.FG).pack(anchor="w", padx=12, pady=(10, 2))
        tk.Label(eig_box, text="Density matrix eigenvalues (descending). Uniform = max entropy.",
                 font=self.FONT_SM, bg="#fff", fg="#666").pack(anchor="w", padx=12, pady=(0, 6))

        self._eig_canvas = tk.Canvas(eig_box, height=100, bg="#fff", highlightthickness=0)
        self._eig_canvas.pack(fill="x", padx=12, pady=(0, 10))

        # ── Interpretation note ───────────────────────────────────────────
        note_box = tk.Frame(r, bg="#fef3c7", relief="solid", bd=1)
        note_box.pack(fill="x")

        tk.Label(note_box, text="INTERPRETATION", font=self.FONT_BOLD,
                 bg="#fef3c7", fg="#92400e").pack(anchor="w", padx=12, pady=(10, 4))
        
        self._note_label = tk.Label(note_box, text="", font=self.FONT_SM,
                                     bg="#fef3c7", fg="#78350f", wraplength=380,
                                     justify="left")
        self._note_label.pack(anchor="w", padx=12, pady=(0, 10))

    # ── State management ──────────────────────────────────────────────────────

    def _populate_cells(self):
        for i in range(self.n):
            for j in range(self.n):
                val = self.matrix[i][j] if i < len(self.matrix) and j < len(self.matrix[i]) else 0.0
                self.cell_vars[i][j].set(f"{val:.3f}")

    def _cell_changed(self, i, j):
        try:
            val = float(self.cell_vars[i][j].get())
        except ValueError:
            return
        self.matrix[i][j] = val
        if self.sym_lock.get() and i != j:
            self.matrix[j][i] = val
            self.cell_vars[j][i].set(f"{val:.3f}")
        self.active_preset.set("")
        self._update_preset_buttons()
        self._recompute()

    def _apply_preset(self, key):
        self.active_preset.set(key)
        if key == "madsen_base":
            self.n = 7
            self.matrix = MADSEN_BASELINE.tolist()
            self.labels = list(NETWORKS)
        elif key == "madsen_peak":
            self.n = 7
            self.matrix = MADSEN_PEAK.tolist()
            self.labels = list(NETWORKS)
        elif key == "pearson":
            self.matrix = pearson_style(self.n).tolist()
            self.labels = [f"R{i+1}" for i in range(self.n)]
        elif key == "identity":
            self.matrix = identity(self.n).tolist()
            self.labels = [f"N{i+1}" for i in range(self.n)]

        self._rebuild_grid()
        self._populate_cells()
        self._update_preset_buttons()
        self._update_size_buttons()
        self._recompute()

    def _change_n(self, new_n):
        self.n = new_n
        self.matrix = identity(new_n).tolist()
        self.labels = [f"N{i+1}" for i in range(new_n)]
        self.active_preset.set("")
        self._rebuild_grid()
        self._populate_cells()
        self._update_preset_buttons()
        self._update_size_buttons()
        self._recompute()

    def _update_preset_buttons(self):
        current = self.active_preset.get()
        for btn in self._preset_btns:
            key = btn._preset_key
            if key == current:
                btn.configure(bg=self.BTN_ACTIVE_BG, fg=self.BTN_ACTIVE_FG)
            else:
                btn.configure(bg=self.BTN_BG, fg=self.BTN_FG)
        
        # Update description
        if current in PRESET_INFO:
            self._preset_desc.config(text=PRESET_INFO[current]["desc"])
        else:
            self._preset_desc.config(text="Custom matrix (edited)")

    def _update_size_buttons(self):
        for btn in self._size_btns:
            v = int(btn.cget("text"))
            if v == self.n:
                btn.configure(bg=self.BTN_ACTIVE_BG, fg=self.BTN_ACTIVE_FG)
            else:
                btn.configure(bg=self.BTN_BG, fg=self.BTN_FG)

    # ── Compute & display ─────────────────────────────────────────────────────

    def _recompute(self):
        result = compute_vne(self.matrix)
        if result is None:
            for lbl in self._readout_labels.values():
                lbl.config(text="ERR")
            self._big_entropy_label.config(text="S/Smax = ERR")
            self._delta_label.config(text="")
            return

        # Update readouts
        self._readout_labels["S / Smax"].config(text=f"{result['normalized_vne']:.4f}")
        self._readout_labels["VNE (nats)"].config(text=f"{result['vne']:.4f}")
        self._readout_labels["Matrix trace"].config(text=f"{result['trace']:.3f}")
        self._readout_labels["Dissolution index"].config(text=f"{result['di']:.4f}")

        # Big display
        self._big_entropy_label.config(text=f"S/Smax = {result['normalized_vne']:.3f}")
        
        # Delta from baseline
        delta = result['normalized_vne'] - BASELINE_REF['normalized_vne']
        if abs(delta) > 0.001:
            color = "#dc2626" if delta < 0 else "#16a34a"
            self._delta_label.config(text=f"(Δ = {delta:+.3f} vs baseline)", fg=color)
        else:
            self._delta_label.config(text="(≈ baseline)")

        # Eigenvalue chart
        self._draw_eigenvalues(result["lambdas"])

        # Interpretation
        nv = result["normalized_vne"]
        preset = self.active_preset.get()
        
        if preset == "madsen_peak":
            note = ("Entropy DECREASED despite dissolution. The visual network (VN = 1.40) "
                    "anchors the trace while association networks erode, collapsing λ₅ and λ₆.")
        elif preset == "pearson":
            note = ("Pearson matrices have unit diagonal by construction—invariant to drug state. "
                    "All perturbation is in off-diagonals, so VNE INCREASES under psychedelics "
                    "(cf. Felippe et al. 2021).")
        elif preset == "madsen_base":
            note = ("Baseline resting-state FC. Note VN diagonal = 1.40 (visual network), "
                    "far exceeding DMN = 0.25. This heterogeneity drives the paradox.")
        elif nv > 0.95:
            note = "Near-uniform eigenvalues → maximum entropy. No dominant spectral mode."
        elif nv > 0.85:
            note = "Moderate uniformity. Eigenvalue spread is relatively flat."
        elif nv > 0.65:
            note = "Some spectral concentration. A few eigenvalues dominate."
        else:
            note = "Strong spectral concentration. One or two eigenvalues capture most variance."
        
        self._note_label.config(text=note)

    def _draw_eigenvalues(self, lambdas):
        c = self._eig_canvas
        c.delete("all")
        self.root.update_idletasks()
        W = c.winfo_width()
        H = c.winfo_height()
        if W < 10 or not lambdas:
            return

        max_l = max(lambdas) if lambdas else 1
        n = len(lambdas)
        pad = 12
        chart_h = H - 28
        bar_w = (W - pad * 2) / n

        # Baseline
        c.create_line(pad, H - 20, W - pad, H - 20, fill="#ccc")

        # Uniform reference line
        uniform = 1.0 / n
        y_uniform = H - 20 - int((uniform / max_l) * chart_h) if max_l > 0 else H - 20
        c.create_line(pad, y_uniform, W - pad, y_uniform, fill="#94a3b8", dash=(4, 4))
        c.create_text(W - pad - 2, y_uniform - 8, text="uniform", font=("Helvetica", 8), 
                      fill="#94a3b8", anchor="e")

        for i, l in enumerate(lambdas):
            x0 = pad + i * bar_w + 3
            x1 = pad + (i + 1) * bar_w - 3
            bh = max(2, int((l / max_l) * chart_h)) if max_l > 0 else 2
            y0 = H - 20 - bh
            y1 = H - 20
            
            # Color: blue if above uniform, gray if below
            fill = self.ACCENT if l >= uniform else "#9ca3af"
            c.create_rectangle(x0, y0, x1, y1, fill=fill, outline="")
            
            xm = (x0 + x1) / 2
            c.create_text(xm, H - 8, text=f"λ{i+1}", font=("Helvetica", 8), fill="#555")
            c.create_text(xm, y0 - 6, text=f"{l:.2f}", font=("Helvetica", 8), fill="#333")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("950x720")
    root.minsize(900, 680)
    app = VNEApp(root)
    root.mainloop()
