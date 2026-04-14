"""
Native GUI Dashboard for Isolation Forest Network Attack Detection.

Tkinter + matplotlib embedded dashboard showing:
  - Confusion matrix (color-coded)
  - 9 classification metrics with gauges
  - Per-class detection report table
  - Model comparison table
  - ROC curve + Precision-Recall curve
  - Score distribution histogram
  - Feature importance chart

Launched automatically by iforest_demo.py after evaluation.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
import numpy as np

# matplotlib with Tk backend
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap


# ── Color Palette ──
BG = "#0a0e1a"
BG2 = "#111827"
CARD = "#1a2235"
CARD_HOVER = "#1f2a40"
BORDER = "#2a3550"
TEXT = "#f1f5f9"
TEXT2 = "#94a3b8"
TEXT_MUTED = "#64748b"
GREEN = "#10b981"
CYAN = "#06b6d4"
RED = "#ef4444"
ORANGE = "#f97316"
YELLOW = "#f59e0b"
BLUE = "#3b82f6"
PURPLE = "#8b5cf6"
PINK = "#ec4899"

PLOT_BG = "#111827"
PLOT_FACE = "#16213e"
PLOT_TEXT = "#eaeaea"
PLOT_GRID = "#333366"

ATTACK_COLORS = {
    "Normal": GREEN,
    "DDoS": RED,
    "Port Scan": ORANGE,
    "Data Exfil": PURPLE,
    "Brute Force": YELLOW,
    "DNS Tunneling": CYAN,
}


def _status_color(val, inverted=False, is_mcc=False):
    if is_mcc:
        return GREEN if val >= 0.7 else (YELLOW if val >= 0.3 else RED)
    if inverted:
        return GREEN if val <= 0.05 else (YELLOW if val <= 0.15 else RED)
    return GREEN if val >= 0.85 else (YELLOW if val >= 0.6 else RED)


def _status_label(val, inverted=False, is_mcc=False):
    c = _status_color(val, inverted, is_mcc)
    if c == GREEN: return "GOOD"
    if c == YELLOW: return "FAIR"
    return "POOR"


class DashboardGUI:
    """Full-screen tkinter dashboard with embedded matplotlib charts."""

    def __init__(self, results: dict):
        """
        Args:
            results: dict with keys:
                tp, tn, fp, fn, accuracy, precision, recall, f1,
                scores, y_true, y_labels, y_pred, threshold, auc,
                model_name, train_time, predict_time,
                n_estimators, psi, contamination,
                feature_names, feature_importances,
                per_class (list of dicts with class, samples, detected, missed, recall),
                model_comparison (list of dicts with name, recall, precision, fnr),
                fprs, tprs (for ROC curve),
                pr_precisions, pr_recalls (for PR curve),
        """
        self.r = results
        self.root = tk.Tk()
        self.root.title("Isolation Forest - ML Classification Dashboard")
        self.root.configure(bg=BG)
        self.root.state("zoomed")  # Maximize on Windows

        self._build_ui()

    def _build_ui(self):
        # Scrollable canvas
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        self.scroll_frame = tk.Frame(canvas, bg=BG)

        self.scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Make content stretch full width
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(
            canvas.find_withtag("all")[0], width=e.width
        ) if canvas.find_withtag("all") else None)

        parent = self.scroll_frame

        # ── Header ──
        self._build_header(parent)

        # ── Row 1: Confusion Matrix + Metrics ──
        row1 = tk.Frame(parent, bg=BG)
        row1.pack(fill=tk.X, padx=20, pady=(10, 5))

        self._build_confusion_matrix(row1)
        self._build_metrics_panel(row1)

        # ── Row 2: Per-class table + Model comparison ──
        row2 = tk.Frame(parent, bg=BG)
        row2.pack(fill=tk.X, padx=20, pady=5)

        self._build_per_class_table(row2)
        self._build_model_comparison(row2)

        # ── Row 3: Charts (2x2) ──
        row3 = tk.Frame(parent, bg=BG)
        row3.pack(fill=tk.X, padx=20, pady=5)

        self._build_score_distribution(row3)
        self._build_roc_curve(row3)

        row4 = tk.Frame(parent, bg=BG)
        row4.pack(fill=tk.X, padx=20, pady=5)

        self._build_pr_curve(row4)
        self._build_feature_importance(row4)

        # ── Footer ──
        self._build_footer(parent)

    # ──────────────────────────────────────────────────────────────
    # HEADER
    # ──────────────────────────────────────────────────────────────

    def _build_header(self, parent):
        header = tk.Frame(parent, bg="#0d1525", pady=16, padx=24)
        header.pack(fill=tk.X)

        title = tk.Label(
            header, text="Isolation Forest for Network Attack Detection",
            font=("Segoe UI", 18, "bold"), fg=BLUE, bg="#0d1525"
        )
        title.pack(side=tk.LEFT)

        badge_frame = tk.Frame(header, bg="#0d2818", padx=12, pady=4,
                               highlightbackground=GREEN, highlightthickness=1)
        badge_frame.pack(side=tk.RIGHT)
        tk.Label(badge_frame, text="MLA Course Project",
                 font=("Segoe UI", 10, "bold"), fg=GREEN, bg="#0d2818").pack()

        sub = tk.Label(
            header,
            text=f"   Model: {self.r.get('model_name', 'iForest')}  |  "
                 f"{self.r.get('n_estimators', 200)} trees  |  "
                 f"psi={self.r.get('psi', 512)}  |  "
                 f"AUC={self.r.get('auc', 0):.4f}",
            font=("Segoe UI", 10), fg=TEXT2, bg="#0d1525"
        )
        sub.pack(side=tk.LEFT, padx=(20, 0))

    # ──────────────────────────────────────────────────────────────
    # CONFUSION MATRIX
    # ──────────────────────────────────────────────────────────────

    def _build_confusion_matrix(self, parent):
        frame = tk.Frame(parent, bg=CARD, padx=16, pady=16,
                         highlightbackground=BORDER, highlightthickness=1)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 8), expand=False)

        tk.Label(frame, text="CONFUSION MATRIX", font=("Segoe UI", 9, "bold"),
                 fg=TEXT_MUTED, bg=CARD).pack(anchor="w", pady=(0, 10))

        fig = Figure(figsize=(3.5, 3), facecolor=CARD)
        ax = fig.add_subplot(111)

        tp, tn, fp, fn = self.r['tp'], self.r['tn'], self.r['fp'], self.r['fn']
        cm = np.array([[tn, fp], [fn, tp]])
        total = tp + tn + fp + fn

        colors = np.array([
            [CYAN, ORANGE],
            [RED, GREEN],
        ])

        # Draw cells manually
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = val / total * 100 if total > 0 else 0
                color = colors[i, j]
                ax.add_patch(plt.Rectangle((j, 1 - i), 0.95, 0.9,
                             facecolor=color, alpha=0.2, edgecolor=color, linewidth=2))
                labels_map = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}
                lbl = labels_map[(i, j)]
                ax.text(j + 0.475, 1.55 - i, f"{lbl}\n{val}\n({pct:.1f}%)",
                        ha='center', va='center', fontsize=11, fontweight='bold',
                        color=color)

        ax.set_xlim(-0.1, 2.1)
        ax.set_ylim(-0.1, 2.1)
        ax.set_xticks([0.475, 1.425])
        ax.set_xticklabels(["Pred Positive", "Pred Negative"], fontsize=8)
        ax.set_yticks([0.5, 1.45])
        ax.set_yticklabels(["Actual Positive", "Actual Negative"], fontsize=8)
        ax.set_facecolor(CARD)
        ax.tick_params(colors=TEXT2)
        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        # Sample counts
        stats = tk.Frame(frame, bg=CARD)
        stats.pack(fill=tk.X, pady=(8, 0))
        for label, val in [("Total", total), ("Pos", tp+fn), ("Neg", fp+tn),
                           ("Prevalence", f"{(tp+fn)/total*100:.1f}%" if total > 0 else "0%")]:
            sf = tk.Frame(stats, bg=CARD)
            sf.pack(side=tk.LEFT, expand=True)
            tk.Label(sf, text=str(val), font=("Consolas", 12, "bold"),
                     fg=TEXT, bg=CARD).pack()
            tk.Label(sf, text=label, font=("Segoe UI", 7), fg=TEXT_MUTED, bg=CARD).pack()

    # ──────────────────────────────────────────────────────────────
    # METRICS PANEL
    # ──────────────────────────────────────────────────────────────

    def _build_metrics_panel(self, parent):
        frame = tk.Frame(parent, bg=CARD, padx=16, pady=16,
                         highlightbackground=BORDER, highlightthickness=1)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(8, 0), expand=True)

        tk.Label(frame, text="CLASSIFICATION METRICS", font=("Segoe UI", 9, "bold"),
                 fg=TEXT_MUTED, bg=CARD).pack(anchor="w", pady=(0, 10))

        tp, tn, fp, fn = self.r['tp'], self.r['tn'], self.r['fp'], self.r['fn']
        total = tp + tn + fp + fn or 1
        ap = tp + fn or 1
        an = fp + tn or 1
        pp = tp + fp or 1

        acc = (tp + tn) / total
        prec = tp / pp
        rec = tp / ap
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        spec = tn / an
        fpr = fp / an
        fnr = fn / ap
        mcc_num = (tp * tn) - (fp * fn)
        mcc_den = np.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)) or 1
        mcc = mcc_num / mcc_den
        bal_acc = (rec + spec) / 2

        metrics = [
            ("Recall *", rec, "TP/(TP+FN)", False, False),
            ("Precision", prec, "TP/(TP+FP)", False, False),
            ("F1 Score", f1, "2PR/(P+R)", False, False),
            ("Accuracy", acc, "(TP+TN)/N", False, False),
            ("Specificity", spec, "TN/(TN+FP)", False, False),
            ("FPR", fpr, "FP/(FP+TN)", True, False),
            ("FNR", fnr, "FN/(TP+FN)", True, False),
            ("MCC", mcc, "Matthews", False, True),
            ("Bal. Accuracy", bal_acc, "(R+S)/2", False, False),
        ]

        grid = tk.Frame(frame, bg=CARD)
        grid.pack(fill=tk.BOTH, expand=True)

        for idx, (name, val, formula, inv, is_mcc) in enumerate(metrics):
            row, col = divmod(idx, 3)
            cell = tk.Frame(grid, bg=BG2, padx=10, pady=8,
                            highlightbackground=BORDER, highlightthickness=1)
            cell.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
            grid.columnconfigure(col, weight=1)
            grid.rowconfigure(row, weight=1)

            color = _status_color(val, inv, is_mcc)
            status = _status_label(val, inv, is_mcc)

            # Top row: name + badge
            top = tk.Frame(cell, bg=BG2)
            top.pack(fill=tk.X)
            tk.Label(top, text=name, font=("Segoe UI", 8, "bold"),
                     fg=TEXT2, bg=BG2).pack(side=tk.LEFT)
            tk.Label(top, text=status, font=("Segoe UI", 7, "bold"),
                     fg=color, bg=BG2).pack(side=tk.RIGHT)

            # Value
            if is_mcc:
                disp = f"{val:.4f}"
            else:
                disp = f"{val*100:.1f}%"
            tk.Label(cell, text=disp, font=("Consolas", 16, "bold"),
                     fg=color, bg=BG2).pack(anchor="w")

            # Gauge bar
            bar_bg = tk.Frame(cell, bg="#0d1321", height=4)
            bar_bg.pack(fill=tk.X, pady=(4, 2))
            bar_bg.pack_propagate(False)
            pct = ((val + 1) / 2) if is_mcc else (1 - val if inv else val)
            pct = max(0, min(1, pct))
            bar_fill = tk.Frame(bar_bg, bg=color, height=4)
            bar_fill.place(relwidth=pct, relheight=1)

            # Formula
            tk.Label(cell, text=formula, font=("Consolas", 7),
                     fg=TEXT_MUTED, bg=BG2).pack(anchor="w")

    # ──────────────────────────────────────────────────────────────
    # PER-CLASS TABLE
    # ──────────────────────────────────────────────────────────────

    def _build_per_class_table(self, parent):
        frame = tk.Frame(parent, bg=CARD, padx=16, pady=16,
                         highlightbackground=BORDER, highlightthickness=1)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 8), expand=True)

        tk.Label(frame, text="PER-CLASS DETECTION", font=("Segoe UI", 9, "bold"),
                 fg=TEXT_MUTED, bg=CARD).pack(anchor="w", pady=(0, 8))

        # Header
        hdr = tk.Frame(frame, bg=BG2)
        hdr.pack(fill=tk.X)
        for col, (text, w) in enumerate([("Class", 16), ("Samples", 8),
                                          ("Detected", 8), ("Missed", 8), ("Recall", 8)]):
            tk.Label(hdr, text=text, font=("Consolas", 8, "bold"), fg=TEXT_MUTED,
                     bg=BG2, width=w, anchor="w").pack(side=tk.LEFT, padx=2)

        per_class = self.r.get("per_class", [])
        for item in per_class:
            row_bg = CARD
            row = tk.Frame(frame, bg=row_bg)
            row.pack(fill=tk.X, pady=1)

            cls = item["class"]
            color = ATTACK_COLORS.get(cls, TEXT)
            rec = item["recall"]
            rec_color = _status_color(rec)

            tk.Label(row, text=cls, font=("Consolas", 9), fg=color,
                     bg=row_bg, width=16, anchor="w").pack(side=tk.LEFT, padx=2)
            tk.Label(row, text=str(item["samples"]), font=("Consolas", 9),
                     fg=TEXT, bg=row_bg, width=8, anchor="w").pack(side=tk.LEFT, padx=2)
            tk.Label(row, text=str(item["detected"]), font=("Consolas", 9),
                     fg=GREEN, bg=row_bg, width=8, anchor="w").pack(side=tk.LEFT, padx=2)
            tk.Label(row, text=str(item["missed"]), font=("Consolas", 9),
                     fg=RED if item["missed"] > 0 else TEXT_MUTED,
                     bg=row_bg, width=8, anchor="w").pack(side=tk.LEFT, padx=2)
            tk.Label(row, text=f"{rec*100:.1f}%", font=("Consolas", 9, "bold"),
                     fg=rec_color, bg=row_bg, width=8, anchor="w").pack(side=tk.LEFT, padx=2)

    # ──────────────────────────────────────────────────────────────
    # MODEL COMPARISON
    # ──────────────────────────────────────────────────────────────

    def _build_model_comparison(self, parent):
        frame = tk.Frame(parent, bg=CARD, padx=16, pady=16,
                         highlightbackground=BORDER, highlightthickness=1)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(8, 0), expand=True)

        tk.Label(frame, text="MODEL COMPARISON", font=("Segoe UI", 9, "bold"),
                 fg=TEXT_MUTED, bg=CARD).pack(anchor="w", pady=(0, 8))

        # Header
        hdr = tk.Frame(frame, bg=BG2)
        hdr.pack(fill=tk.X)
        for text, w in [("Model", 24), ("Recall", 8), ("Prec", 8), ("FNR", 8)]:
            tk.Label(hdr, text=text, font=("Consolas", 8, "bold"), fg=TEXT_MUTED,
                     bg=BG2, width=w, anchor="w").pack(side=tk.LEFT, padx=2)

        comparisons = self.r.get("model_comparison", [])
        best_name = self.r.get("model_name", "")

        for item in comparisons:
            is_best = item["name"] == best_name
            row_bg = "#1a2a1a" if is_best else CARD
            row = tk.Frame(frame, bg=row_bg)
            row.pack(fill=tk.X, pady=1)

            name_text = item["name"]
            if is_best:
                name_text += " *"

            tk.Label(row, text=name_text, font=("Consolas", 9, "bold" if is_best else ""),
                     fg=GREEN if is_best else TEXT, bg=row_bg, width=24,
                     anchor="w").pack(side=tk.LEFT, padx=2)
            tk.Label(row, text=f"{item['recall']*100:.1f}%", font=("Consolas", 9, "bold"),
                     fg=_status_color(item['recall']), bg=row_bg, width=8,
                     anchor="w").pack(side=tk.LEFT, padx=2)
            tk.Label(row, text=f"{item['precision']*100:.1f}%", font=("Consolas", 9),
                     fg=TEXT, bg=row_bg, width=8, anchor="w").pack(side=tk.LEFT, padx=2)
            tk.Label(row, text=f"{item['fnr']*100:.1f}%", font=("Consolas", 9, "bold"),
                     fg=_status_color(item['fnr'], inverted=True), bg=row_bg, width=8,
                     anchor="w").pack(side=tk.LEFT, padx=2)

        # IDS note
        note = tk.Frame(frame, bg="#1a1020", padx=8, pady=6)
        note.pack(fill=tk.X, pady=(10, 0))
        tk.Label(note, text="IDS Priority: Recall > Precision",
                 font=("Segoe UI", 8, "bold"), fg=RED, bg="#1a1020").pack(anchor="w")
        tk.Label(note, text="Missing attacks (FN) is worse than false alarms (FP)",
                 font=("Segoe UI", 7), fg=TEXT_MUTED, bg="#1a1020").pack(anchor="w")

    # ──────────────────────────────────────────────────────────────
    # CHARTS
    # ──────────────────────────────────────────────────────────────

    def _make_chart_frame(self, parent, title):
        frame = tk.Frame(parent, bg=CARD, padx=12, pady=12,
                         highlightbackground=BORDER, highlightthickness=1)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=4, pady=4, expand=True)
        tk.Label(frame, text=title, font=("Segoe UI", 9, "bold"),
                 fg=TEXT_MUTED, bg=CARD).pack(anchor="w", pady=(0, 4))
        return frame

    def _build_score_distribution(self, parent):
        frame = self._make_chart_frame(parent, "SCORE DISTRIBUTION")
        fig = Figure(figsize=(5, 3), facecolor=CARD)
        ax = fig.add_subplot(111)

        scores = self.r.get("scores", np.array([]))
        y_true = self.r.get("y_true", np.array([]))
        threshold = self.r.get("threshold", 0.5)

        if len(scores) > 0:
            normal_s = scores[y_true == 0]
            attack_s = scores[y_true == 1]
            ax.hist(normal_s, bins=50, alpha=0.7, color=GREEN,
                    label=f"Normal ({len(normal_s)})", edgecolor='none')
            ax.hist(attack_s, bins=50, alpha=0.7, color=RED,
                    label=f"Attack ({len(attack_s)})", edgecolor='none')
            ax.axvline(x=threshold, color=YELLOW, linestyle='--', linewidth=2,
                       label=f"Threshold ({threshold:.3f})")
            ax.legend(fontsize=7, facecolor=PLOT_FACE, edgecolor=PLOT_GRID, labelcolor=PLOT_TEXT)

        ax.set_xlabel("Anomaly Score", fontsize=8, color=PLOT_TEXT)
        ax.set_ylabel("Frequency", fontsize=8, color=PLOT_TEXT)
        ax.set_facecolor(PLOT_FACE)
        ax.tick_params(colors=TEXT2, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(PLOT_GRID)
        ax.grid(True, alpha=0.2, color=PLOT_GRID)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_roc_curve(self, parent):
        frame = self._make_chart_frame(parent, f"ROC CURVE (AUC={self.r.get('auc', 0):.4f})")
        fig = Figure(figsize=(5, 3), facecolor=CARD)
        ax = fig.add_subplot(111)

        fprs = self.r.get("fprs", np.array([]))
        tprs = self.r.get("tprs", np.array([]))
        auc_val = self.r.get("auc", 0)

        if len(fprs) > 0:
            ax.plot(fprs, tprs, color=BLUE, linewidth=2,
                    label=f"iForest (AUC={auc_val:.4f})")
            ax.fill_between(fprs, tprs, alpha=0.15, color=BLUE)
        ax.plot([0, 1], [0, 1], color=TEXT_MUTED, linestyle='--', linewidth=1,
                label="Random")
        ax.legend(fontsize=7, facecolor=PLOT_FACE, edgecolor=PLOT_GRID, labelcolor=PLOT_TEXT)

        # Mark operating point
        tp, fn, fp, tn = self.r['tp'], self.r['fn'], self.r['fp'], self.r['tn']
        op_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        op_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        ax.plot(op_fpr, op_tpr, 'o', color=YELLOW, markersize=8, zorder=5)
        ax.annotate(f"({op_fpr:.2f}, {op_tpr:.2f})", (op_fpr, op_tpr),
                    textcoords="offset points", xytext=(10, -10),
                    fontsize=7, color=YELLOW)

        ax.set_xlabel("False Positive Rate", fontsize=8, color=PLOT_TEXT)
        ax.set_ylabel("True Positive Rate", fontsize=8, color=PLOT_TEXT)
        ax.set_facecolor(PLOT_FACE)
        ax.tick_params(colors=TEXT2, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(PLOT_GRID)
        ax.grid(True, alpha=0.2, color=PLOT_GRID)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_pr_curve(self, parent):
        frame = self._make_chart_frame(parent, "PRECISION-RECALL CURVE")
        fig = Figure(figsize=(5, 3), facecolor=CARD)
        ax = fig.add_subplot(111)

        pr_p = self.r.get("pr_precisions", np.array([]))
        pr_r = self.r.get("pr_recalls", np.array([]))

        if len(pr_p) > 0:
            ax.plot(pr_r, pr_p, color=PURPLE, linewidth=2, label="iForest")
            ax.fill_between(pr_r, pr_p, alpha=0.15, color=PURPLE)

        # Operating point
        prec = self.r.get("precision", 0)
        rec = self.r.get("recall", 0)
        ax.plot(rec, prec, 'o', color=YELLOW, markersize=8, zorder=5)
        ax.annotate(f"(R={rec:.2f}, P={prec:.2f})", (rec, prec),
                    textcoords="offset points", xytext=(10, -10),
                    fontsize=7, color=YELLOW)

        ax.legend(fontsize=7, facecolor=PLOT_FACE, edgecolor=PLOT_GRID, labelcolor=PLOT_TEXT)
        ax.set_xlabel("Recall", fontsize=8, color=PLOT_TEXT)
        ax.set_ylabel("Precision", fontsize=8, color=PLOT_TEXT)
        ax.set_facecolor(PLOT_FACE)
        ax.tick_params(colors=TEXT2, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(PLOT_GRID)
        ax.grid(True, alpha=0.2, color=PLOT_GRID)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_feature_importance(self, parent):
        frame = self._make_chart_frame(parent, "FEATURE IMPORTANCE")
        fig = Figure(figsize=(5, 3), facecolor=CARD)
        ax = fig.add_subplot(111)

        names = self.r.get("feature_names", [])
        importances = self.r.get("feature_importances", np.array([]))

        if len(importances) > 0:
            sorted_idx = np.argsort(importances)
            sorted_names = [names[i] for i in sorted_idx]
            sorted_vals = importances[sorted_idx]

            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
            bars = ax.barh(range(len(names)), sorted_vals, color=colors, height=0.7)

            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(sorted_names, fontsize=6)

            for bar, val in zip(bars, sorted_vals):
                ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va='center', fontsize=6, color=PLOT_TEXT)

        ax.set_xlabel("Importance", fontsize=8, color=PLOT_TEXT)
        ax.set_facecolor(PLOT_FACE)
        ax.tick_params(colors=TEXT2, labelsize=6)
        for spine in ax.spines.values():
            spine.set_color(PLOT_GRID)
        ax.grid(True, alpha=0.2, axis='x', color=PLOT_GRID)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ──────────────────────────────────────────────────────────────
    # FOOTER
    # ──────────────────────────────────────────────────────────────

    def _build_footer(self, parent):
        footer = tk.Frame(parent, bg=BG, pady=12)
        footer.pack(fill=tk.X, padx=20)

        info = (
            f"Train: {self.r.get('train_time', 0):.2f}s  |  "
            f"Predict: {self.r.get('predict_time', 0):.2f}s  |  "
            f"Pure-Python (NumPy only)  |  "
            f"Semester 2026"
        )
        tk.Label(footer, text=info, font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG).pack()

    # ──────────────────────────────────────────────────────────────
    # RUN
    # ──────────────────────────────────────────────────────────────

    def show(self):
        """Display the dashboard (blocking)."""
        self.root.mainloop()


class LaunchWindow:
    """Pre-dashboard loading window that captures stdout."""
    def __init__(self, run_func):
        self.run_func = run_func
        self.root = tk.Tk()
        self.root.title("Isolation Forest - Running Demo...")
        self.root.geometry("800x600")
        self.root.configure(bg=BG)

        # Center window
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"{w}x{h}+{x}+{y}")

        # Header
        hdr = tk.Frame(self.root, bg="#0d1525", pady=16)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="Running ML Pipeline...", font=("Segoe UI", 16, "bold"),
                 fg=BLUE, bg="#0d1525").pack()

        # Terminal
        term_frame = tk.Frame(self.root, bg=BG2, padx=10, pady=10)
        term_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.text = tk.Text(term_frame, bg=BG2, fg=GREEN, font=("Consolas", 10),
                            wrap=tk.WORD, borderwidth=0, highlightthickness=0)
        self.text.pack(fill=tk.BOTH, expand=True)

        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode="indeterminate")
        self.progress.pack(fill=tk.X, padx=20, pady=(0, 20))

        # Redirect stdout
        import sys
        class StdoutRedirect:
            def __init__(self, text_widget):
                self.text_space = text_widget
                self.sys_stdout = sys.stdout
            def write(self, string):
                self.text_space.insert(tk.END, string)
                self.text_space.see(tk.END)
                self.sys_stdout.write(string)
                self.text_space.update_idletasks()
            def flush(self):
                self.sys_stdout.flush()

        self._sys_stdout_orig = sys.stdout
        sys.stdout = StdoutRedirect(self.text)

    def start(self):
        self.progress.start(10)
        import threading
        t = threading.Thread(target=self._run_bg)
        t.daemon = True
        t.start()
        self.root.mainloop()

    def _run_bg(self):
        try:
            results = self.run_func()
            self.root.after(1000, lambda: self._on_complete(results))
        except Exception as e:
            import sys, traceback
            print(f"\n[!] Error: {e}")
            traceback.print_exc()
            self.progress.stop()

    def _on_complete(self, results):
        import sys
        sys.stdout = self._sys_stdout_orig
        self.root.destroy()
        if results:
            launch_dashboard(results)


def launch_dashboard(results: dict):
    """Convenience entry point for the main dashboard."""
    gui = DashboardGUI(results)
    gui.show()

def run_demo_with_gui(run_func):
    """Run a long function with a loading GUI, then launch the dashboard."""
    loader = LaunchWindow(run_func)
    loader.start()

