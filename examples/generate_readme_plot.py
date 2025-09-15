#!/usr/bin/env python3
"""
Generate the Quick Start figure and save to assets/readme_quickstart.png
Uses a non-interactive backend to avoid PTY/GUI issues.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from trendfilter import TrendFilter, CVTrendFilter
from matplotlib import colors
from matplotlib.ticker import LogFormatter, ScalarFormatter


def main() -> str:
    repo_root = os.path.dirname(os.path.dirname(__file__))
    out_dir = os.path.join(repo_root, 'assets')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'readme_quickstart.png')

    # Generate sample data
    n = 100
    x = np.linspace(0, 1, n)
    # Original base curve, then add an extra sine only on the second segment
    break_pt = 0.5
    base_signal = np.sin(2 * np.pi * x) + 0.5 * np.cos(4 * np.pi * x)
    # Extra sine (zero at break to keep value continuity): A * sin(2π f (x - break_pt)) for x >= break_pt
    A_extra, f_extra = 1.2, 6.0
    extra = A_extra * np.sin(2 * np.pi * f_extra * (x - break_pt))
    true_signal = np.where(x < break_pt, base_signal, base_signal + extra)
    y = true_signal + 0.03 * np.random.randn(n)

    # Fit trend filter
    tf = TrendFilter(order=2, lambda_reg=0.1)
    tf.fit(y)
    y_fit = tf.predict()

    # Cross-validation for parameter selection
    cv_tf = CVTrendFilter(order=2)
    cv_tf.fit(y)
    y_fit_cv = cv_tf.predict()

    # Build long-form dataframes with lambda columns
    lam_basic = getattr(tf, 'lambda_', None)
    if lam_basic is None:
        lam_basic = np.arange(y_fit.shape[1], dtype=float)
    df_basic = pd.DataFrame(y_fit, columns=np.round(lam_basic, 8))
    df_basic['x'] = x
    dfb = df_basic.melt(id_vars='x', var_name='lambda', value_name='y_hat')

    lam_cv = getattr(cv_tf, 'lambda_path_', None)
    if lam_cv is None:
        lam_cv = np.arange(y_fit_cv.shape[1], dtype=float)
    df_cv = pd.DataFrame(y_fit_cv, columns=np.round(lam_cv, 8))
    df_cv['x'] = x
    dfcv = df_cv.melt(id_vars='x', var_name='lambda', value_name='y_hat')

    # Plot
    plt.figure(figsize=(18, 5))

    # Helper to choose normalization (log if wide positive range)
    def choose_norm(vals: np.ndarray):
        vals = np.asarray(vals, dtype=float)
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        if np.isfinite(vmin) and np.isfinite(vmax) and vmin > 0 and vmax / max(vmin, 1e-12) >= 50:
            return colors.LogNorm(vmin=vmin, vmax=vmax), 'log'
        return colors.Normalize(vmin=vmin, vmax=vmax), 'linear'

    # Compute knot indices from alpha (preferred) or from (order+1)-th finite difference of coef
    def compute_knots(order: int, coef_col: np.ndarray, alpha_col: np.ndarray | None) -> np.ndarray:
        try:
            if alpha_col is not None and alpha_col.size > 0:
                a = np.asarray(alpha_col, dtype=float)
                thr = 1e-6 * np.nanmax(np.abs(a)) if np.nanmax(np.abs(a)) > 0 else 0.0
                idx = np.where(np.abs(a) > thr)[0]
                return idx + (order + 1)
        except Exception:
            pass
        c = np.asarray(coef_col, dtype=float)
        d = np.diff(c, n=order + 1)
        thr = 1e-6 * np.nanmax(np.abs(d)) if np.nanmax(np.abs(d)) > 0 else 0.0
        idx = np.where(np.abs(d) > thr)[0]
        return idx + (order + 1)

    # Panel 1: Basic TF — one curve per lambda
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(x, y, 'o', alpha=0.25, label='Noisy data')
    ax1.plot(x, true_signal, '-', linewidth=2, label='True signal', color='black')
    lam_basic_vals = dfb['lambda'].astype(float).values
    cmap_b = plt.get_cmap('viridis')
    if np.unique(lam_basic_vals).size > 1:
        norm_b, scale_b = choose_norm(lam_basic_vals)
        for lam, grp in dfb.groupby('lambda'):
            lam_f = float(lam)
            ax1.plot(grp['x'].values, grp['y_hat'].values,
                     color=cmap_b(norm_b(lam_f)), alpha=0.35, linewidth=1)
        from matplotlib.cm import ScalarMappable
        mappable_b = ScalarMappable(norm=norm_b, cmap=cmap_b)
        cbar_b = plt.colorbar(mappable_b, ax=ax1, fraction=0.046, pad=0.04)
        cbar_b.set_label('λ (basic)')
        cbar_b.formatter = LogFormatter() if scale_b == 'log' else ScalarFormatter(useMathText=True)
        cbar_b.update_ticks()
    else:
        lam = float(lam_basic_vals[0]) if lam_basic_vals.size else float('nan')
        ax1.plot(x, y_fit[:, 0] if y_fit.ndim == 2 else y_fit, color=cmap_b(0.7), alpha=0.8, linewidth=1.5,
                 label=f'λ={lam:.4g}')
        ax1.legend(loc='best', fontsize=8)
    # Draw knots for a representative basic λ (middle index if multiple)
    try:
        if hasattr(tf, 'coef_'):
            coef2d = tf.coef_ if np.ndim(tf.coef_) == 2 else tf.coef_[:, None]
            alpha2d = getattr(tf, 'alpha_', None)
            if alpha2d is not None and np.ndim(alpha2d) == 1:
                alpha2d = alpha2d[:, None]
            idx_sel = coef2d.shape[1] // 2
            sel_alpha = alpha2d[:, idx_sel] if (alpha2d is not None and alpha2d.shape[1] > idx_sel) else None
            knots_idx = compute_knots(tf.order, coef2d[:, idx_sel], sel_alpha)
            if knots_idx.size:
                x_knots = x[knots_idx]
                lam_val = None
                if hasattr(tf, 'lambda_') and len(tf.lambda_) > idx_sel:
                    try:
                        lam_val = float(tf.lambda_[idx_sel])
                    except Exception:
                        lam_val = None
                print("Basic TF knots:")
                print(f"  lambda_index={idx_sel}" + (f", lambda={lam_val:.6g}" if lam_val is not None else ""))
                print(f"  idx={knots_idx.tolist()}")
                print(f"  x={x_knots.tolist()}")
                for j, xv in enumerate(x_knots):
                    ax1.axvline(float(xv), color='red', linestyle='--', alpha=0.35, linewidth=1,
                                label='knots' if j == 0 else None)
                # ensure a single legend entry exists
                handles, labels = ax1.get_legend_handles_labels()
                if 'knots' not in labels:
                    ax1.plot([], [], color='red', linestyle='--', label='knots')
            else:
                print("Basic TF knots: none")
    except Exception as e:
        print(f"[warn] basic knots computation failed: {e}")

    ax1.set_title('Basic TF: one curve per λ')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # Panel 2: CV TF — one curve per lambda
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(x, y, 'o', alpha=0.25, label='Noisy data')
    ax2.plot(x, true_signal, '-', linewidth=2, label='True signal', color='black')
    lam_unique_cv = np.sort(dfcv['lambda'].astype(float).unique())
    cmap_c = plt.get_cmap('viridis')
    if lam_unique_cv.size > 1:
        norm_c, scale_c = choose_norm(lam_unique_cv)
    else:
        norm_c, scale_c = colors.Normalize(vmin=float(lam_unique_cv[0]), vmax=float(lam_unique_cv[0])), 'linear'
    best_lam = getattr(cv_tf, 'best_lambda_', None)
    for lam, grp in dfcv.groupby('lambda'):
        lam_f = float(lam)
        is_best = best_lam is not None and np.isclose(lam_f, float(best_lam))
        lw = 2 if is_best else 0.8
        alpha = 0.9 if is_best else 0.25
        ax2.plot(grp['x'].values, grp['y_hat'].values,
                 color=cmap_c(norm_c(lam_f)), alpha=alpha, linewidth=lw)
    from matplotlib.cm import ScalarMappable
    mappable_c = ScalarMappable(norm=norm_c, cmap=cmap_c)
    cbar_c = plt.colorbar(mappable_c, ax=ax2, fraction=0.046, pad=0.04)
    cbar_c.set_label('λ (CV)')
    cbar_c.formatter = LogFormatter() if scale_c == 'log' else ScalarFormatter(useMathText=True)
    cbar_c.update_ticks()
    if best_lam is not None:
        ax2.set_title(f'CV TF: one curve per λ (best λ={best_lam:.4f})')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    # Draw knots for best CV λ if available
    try:
        if hasattr(cv_tf, 'best_estimator_') and hasattr(cv_tf.best_estimator_, 'coef_'):
            te = cv_tf.best_estimator_
            coef2d = te.coef_ if np.ndim(te.coef_) == 2 else te.coef_[:, None]
            alpha2d = getattr(te, 'alpha_', None)
            if alpha2d is not None and np.ndim(alpha2d) == 1:
                alpha2d = alpha2d[:, None]
            idx_sel = 0
            sel_alpha = alpha2d[:, idx_sel] if (alpha2d is not None and alpha2d.shape[1] > idx_sel) else None
            knots_idx = compute_knots(te.order, coef2d[:, idx_sel], sel_alpha)
            if knots_idx.size:
                x_knots = x[knots_idx]
                lam_val = getattr(cv_tf, 'best_lambda_', None)
                print("CV TF knots (best λ):")
                if lam_val is not None:
                    print(f"  lambda={float(lam_val):.6g}")
                print(f"  idx={knots_idx.tolist()}")
                print(f"  x={x_knots.tolist()}")
                for j, xv in enumerate(x_knots):
                    ax2.axvline(float(xv), color='magenta', linestyle='--', alpha=0.35, linewidth=1,
                                label='knots (best)' if j == 0 else None)
                handles, labels = ax2.get_legend_handles_labels()
                if 'knots (best)' not in labels:
                    ax2.plot([], [], color='magenta', linestyle='--', label='knots (best)')
            else:
                lam_val = getattr(cv_tf, 'best_lambda_', None)
                if lam_val is not None:
                    print(f"CV TF knots (best λ={float(lam_val):.6g}): none")
                else:
                    print("CV TF knots: none")
        # Additionally, compute knots at a specified lambda (e.g., 8.27) if requested
        chosen_lambda = 8.27
        tf_alt = TrendFilter(order=cv_tf.order, lambda_reg=chosen_lambda)
        tf_alt.fit(y, x)
        coef2d_alt = tf_alt.coef_ if np.ndim(tf_alt.coef_) == 2 else tf_alt.coef_[:, None]
        alpha2d_alt = getattr(tf_alt, 'alpha_', None)
        if alpha2d_alt is not None and np.ndim(alpha2d_alt) == 1:
            alpha2d_alt = alpha2d_alt[:, None]
        sel_alpha_alt = alpha2d_alt[:, 0] if (alpha2d_alt is not None) else None
        knots_idx_alt = compute_knots(tf_alt.order, coef2d_alt[:, 0], sel_alpha_alt)
        if knots_idx_alt.size:
            x_knots_alt = x[knots_idx_alt]
            print("CV TF knots (λ=8.27, manual):")
            print(f"  idx={knots_idx_alt.tolist()}")
            print(f"  x={x_knots_alt.tolist()}")
            for j, xv in enumerate(x_knots_alt):
                ax2.axvline(float(xv), color='orange', linestyle='--', alpha=0.5, linewidth=1,
                            label='knots (λ=8.27)' if j == 0 else None)
            handles, labels = ax2.get_legend_handles_labels()
            if 'knots (λ=8.27)' not in labels:
                ax2.plot([], [], color='orange', linestyle='--', label='knots (λ=8.27)')
        else:
            print("CV TF knots (λ=8.27, manual): none")
    except Exception as e:
        print(f"[warn] CV knots computation failed: {e}")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    return out_path


if __name__ == '__main__':
    path = main()
    print(f'Saved figure to: {path}')
