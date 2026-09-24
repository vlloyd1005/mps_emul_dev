"""
eval_ntrain_scaling.py — N-train Scaling Evaluation Script

Loads each trained emulator model (varying n_batches x cosmo_type), evaluates
predictions against true CAMB power spectra on the test set, and produces:

  1. A line plot of the 95th-percentile fractional error at k~1 h/Mpc vs N_train
  2. Per-model emulator error band plots
  3. Per-model Syren (EH approximation) error band plots

Called automatically by train_all_parallel.sh once all training jobs complete.

Usage (standalone):
    python ./mps_emu/eval_ntrain_scaling.py \
        --prior_type constrained \
        --nl_type lin

Author: Victoria Lloyd (2026)
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import emulmps_w0wa as pk_emu
import train_utils_pk_emulator_v3 as utils


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate emulator N-train scaling and produce summary plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prior_type", default="constrained",
                        choices=["constrained", "expanded"])
    parser.add_argument("--nl_type", default="lin",
                        choices=["lin", "halofit", "mead2020", "mead2020_feedback"])
    parser.add_argument("--test_batch", type=int, default=100,
                        help="Batch index used as the test/validation set.")
    parser.add_argument("--iz", type=int, default=0,
                        help="Redshift index to evaluate at (default 0 → z=0).")
    parser.add_argument("--fig_dir",
                        default="/gpfs/projects/MirandaGroup/victoria/cocoa/Cocoa/mps_emu/validation_figs",
                        help="Directory to save output figures.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DVS_PER_BATCH  = 2000
N_BATCHES_LIST = [5, 25, 50, 100, 150, 200, 250] #[5, 10, 15, 20, 30, 40, 50]

COSMO_CONFIGS = {
    "w0wacdm": {"label": r"$w_0 w_a$CDM", "marker": "o", "color": "#2166ac"},
    "lcdm":    {"label": r"$\Lambda$CDM",  "marker": "s", "color": "#d6604d"},
}

# Plotting style (matching existing eval script)
COLOR50   = "#473C8A"
COLOR90   = "#C45858"
COLOR100  = "lightgray"
AXES_FS   = 20
TICK_FS   = 17
LEGEND_FS = 17
ANNOT_FS  = 12

# Legend handles for percentile bands
HANDLE_98  = mlines.Line2D([], [], color=COLOR100, label=r"$98\%$")
HANDLE_90  = mpatches.Patch(facecolor=COLOR90, label=r"$90\%$")
HANDLE_50  = mpatches.Patch(facecolor=COLOR50, hatch="\\", edgecolor="lightgray", label=r"$50\%$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_k_index(ks, k_target=1.0):
    """Return the index of the k value closest to k_target."""
    return int(np.argmin(np.abs(ks - k_target)))


def _percentiles(err):
    """Return (p2, p10, p25, p75, p90, p98) over axis=0 (across cosmologies)."""
    ps = np.percentile(err, [2, 10, 25, 75, 90, 98], axis=0)
    return ps[0], ps[1], ps[2], ps[3], ps[4], ps[5]


def _fill_ax(ax, ks, p2, p10, p25, p75, p90, p98, ylabel, label_text):
    """Draw percentile bands on ax (matching existing eval script style)."""
    ax.semilogx(ks, p2,  color=COLOR100)
    ax.semilogx(ks, p98, color=COLOR100)
    ax.fill_between(ks, p10, p90, color=COLOR90, alpha=0.7)
    ax.fill_between(ks, p25, p75, color=COLOR50, hatch="\\",
                    edgecolor="lightgray", alpha=0.7)
    ax.set_ylabel(ylabel, fontsize=AXES_FS)
    ax.set_xscale("log")
    ax.grid(alpha=0.4)
    ax.set_xlim([ks[0], ks[-1]])
    ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
    ax.text(0.05, 0.95, label_text, transform=ax.transAxes,
            ha='left', va='top', fontsize=AXES_FS)


def get_pred_and_syren(cosmo_type, prior_type, nl_type, n_batches, test_set, iz):
    """
    Run the emulator and EH approximation for all cosmologies in test_set.

    Returns
    -------
    pred_pks   : (N_valid, N_k) — full emulator P(k)
    syren_pks  : (N_valid, N_k) — EH-only approximation P(k)
    true_pks   : (N_valid, N_k) — CAMB ground truth P(k)
    valid_mask : (N_cosmo,) bool — which cosmologies had no NaN
    """
    true_pks_all = test_set.pks_lin[:, iz, :]

    pred_list  = []
    syren_list = []

    for params in test_set.lhs:
        _, _, pk_full   = pk_emu.get_pks(params, cosmo_type=cosmo_type,
                                          prior_type=prior_type, nl_type=nl_type,
                                          num_batches=n_batches,
                                          use_approximation_only=False)
        _, _, pk_approx = pk_emu.get_pks(params, cosmo_type=cosmo_type,
                                          prior_type=prior_type, nl_type=nl_type,
                                          num_batches=n_batches,
                                          use_approximation_only=True)
        pred_list.append(pk_full[iz])
        syren_list.append(pk_approx[iz])

    pred_pks  = np.array(pred_list)
    syren_pks = np.array(syren_list)

    # Remove rows with NaN (use syren NaN mask as proxy — same k grid)
    nan_mask   = np.isnan(syren_pks).any(axis=1) | np.isnan(pred_pks).any(axis=1)
    valid_mask = ~nan_mask

    if nan_mask.any():
        print(f"    WARNING: {nan_mask.sum()} cosmologies returned NaN — dropping.")

    return (pred_pks[valid_mask], syren_pks[valid_mask],
            true_pks_all[valid_mask], valid_mask)


def compute_p95_at_k1(errors, ks):
    """
    Return the 95th percentile of |error| evaluated at the k bin closest to
    k=1 h/Mpc, taken across all cosmologies.
    """
    ik = _find_k_index(ks, k_target=1.0)
    abs_err_at_k1 = np.abs(errors[:, ik])   # (N_cosmo,)
    return float(np.percentile(abs_err_at_k1, 95)), ik


# ---------------------------------------------------------------------------
# Plot: error band figure for a single (cosmo, n_batches) model
# ---------------------------------------------------------------------------

def make_error_band_plots(errors_emul, errors_syren, ks, iz, z_val,
                           cosmo_type, prior_type, nl_type, n_batches,
                           fig_dir):
    """
    Save one 2-panel figure for the emulator and one for Syren,
    each showing the full-dataset percentile bands (no w0 filtering).
    """
    n_dvs   = n_batches * DVS_PER_BATCH
    z_label = f"z={z_val:.2f}"
    tag     = f"{cosmo_type}_{prior_type}_{nl_type}_nTrain{n_batches}"

    p_emu = _percentiles(errors_emul)
    p_syr = _percentiles(errors_syren)

    for (percs, ylabel, name) in [
        (p_emu,
         r"$P(k)_\mathrm{EmulMPS}/P(k)_\mathrm{CAMB} - 1$",
         "emul"),
        (p_syr,
         r"$P(k)_\mathrm{Syren}/P(k)_\mathrm{CAMB} - 1$",
         "syren"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        _fill_ax(ax, ks, *percs, ylabel=ylabel,
                 label_text=fr"${z_label}$  |  {cosmo_type}  |  $N_\mathrm{{train}}={n_dvs:,}$")
        ax.set_xlabel(r"$k \; (h/\mathrm{Mpc})$", fontsize=AXES_FS)
        ax.legend(handles=[HANDLE_50, HANDLE_90, HANDLE_98],
                  fontsize=LEGEND_FS, loc="lower right")
        plt.tight_layout()
        fname = f"{name}_errors_{tag}_iz{iz}.pdf"
        plt.savefig(os.path.join(fig_dir, fname), bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family']      = 'STIXGeneral'

    z_val   = utils.z_mps[args.iz]
    z_label = f"z={z_val:.2f}"

    print("=" * 60)
    print("[INFO] N-train Scaling Evaluation")
    print(f"       prior_type = {args.prior_type}")
    print(f"       nl_type    = {args.nl_type}")
    print(f"       test_batch = {args.test_batch}")
    print(f"       eval at    = {z_label}  (iz={args.iz})")
    print("=" * 60)

    # --- Load test sets ---
    print("\n[INFO] Loading test sets...")
    test_sets = {}
    for cosmo_type in COSMO_CONFIGS:
        print(f"  Loading {cosmo_type}...")
        test_sets[cosmo_type] = utils.COLASet(
            target_z=utils.z_mps,
            cosmo_type=cosmo_type,
            prior_type=args.prior_type,
            nl_type=args.nl_type,
            start_batch=args.test_batch,
            n_batches=1,
        )

    ks = test_sets["w0wacdm"].ks
    ik1 = _find_k_index(ks, k_target=1.0)
    print(f"\n[INFO] k=1 h/Mpc closest bin: k[{ik1}] = {ks[ik1]:.4f} h/Mpc")

    # --- Evaluate all (cosmo, n_batches) combinations ---
    print("\n[INFO] Evaluating all models...\n")
    results_emul  = {cosmo: [] for cosmo in COSMO_CONFIGS}
    results_syren = {cosmo: [] for cosmo in COSMO_CONFIGS}

    for cosmo_type, test_set in test_sets.items():
        print(f"[INFO] cosmo_type = {cosmo_type}")
        for n_batches in N_BATCHES_LIST:
            n_dvs = n_batches * DVS_PER_BATCH
            print(f"  [n_batches={n_batches:2d}  ({n_dvs:,} DVs)]  evaluating...",
                  flush=True)

            try:
                pred_pks, syren_pks, true_pks, _ = get_pred_and_syren(
                    cosmo_type, args.prior_type, args.nl_type,
                    n_batches, test_set, args.iz
                )
            except Exception as exc:
                print(f"    WARNING: evaluation failed — {exc}")
                results_emul[cosmo_type].append(np.nan)
                results_syren[cosmo_type].append(np.nan)
                continue

            errors_emul  = pred_pks  / true_pks - 1   # (N, N_k)
            errors_syren = syren_pks / true_pks - 1

            # 95th pct of |error| at k~1
            p95_emul,  _ = compute_p95_at_k1(errors_emul,  ks)
            p95_syren, _ = compute_p95_at_k1(errors_syren, ks)

            print(f"    → 95th pct |error| at k=1:  emul={p95_emul:.6f}  "
                  f"syren={p95_syren:.6f}  "
                  f"(n_valid={len(pred_pks)}/{len(test_set.lhs)})")

            results_emul[cosmo_type].append(p95_emul)
            results_syren[cosmo_type].append(p95_syren)

            # Save per-model error band figures
            make_error_band_plots(
                errors_emul, errors_syren, ks,
                args.iz, z_val,
                cosmo_type, args.prior_type, args.nl_type, n_batches,
                args.fig_dir
            )

        print()

    n_dvs_list = [nb * DVS_PER_BATCH for nb in N_BATCHES_LIST]

    # --- Print summary table ---
    print("=" * 70)
    print(f"[INFO] Summary — 95th pct |error| at k={ks[ik1]:.3f} h/Mpc")
    header = f"  {'N_DVs':<12s}"
    for cosmo in COSMO_CONFIGS:
        header += f"  {'emul_'+cosmo:<18s}  {'syren_'+cosmo:<18s}"
    print(header)
    print("  " + "-" * 66)
    for i, nd in enumerate(n_dvs_list):
        row = f"  {nd:<12,d}"
        for cosmo in COSMO_CONFIGS:
            ve = results_emul[cosmo][i]
            vs = results_syren[cosmo][i]
            row += f"  {ve:<18.6f}  {vs:<18.6f}" if not np.isnan(ve) else f"  {'FAILED':<18s}  {'FAILED':<18s}"
        print(row)
    print("=" * 70)

    # --- Scaling plot (emulator only) ---
    fig, ax = plt.subplots(figsize=(8, 6))

    for cosmo_type, cfg in COSMO_CONFIGS.items():
        p95_vals = np.array(results_emul[cosmo_type])
        valid    = ~np.isnan(p95_vals)
        x_plot   = np.array(n_dvs_list)[valid]
        y_plot   = p95_vals[valid]

        ax.plot(x_plot, y_plot,
                color=cfg["color"], marker=cfg["marker"],
                markersize=9, linewidth=2, label=cfg["label"], zorder=3)

        for x, y in zip(x_plot, y_plot):
            ax.annotate(f"{y:.4f}", xy=(x, y), xytext=(0, 10),
                        textcoords="offset points", ha="center",
                        fontsize=ANNOT_FS, color=cfg["color"])

    ax.set_xlabel(r"$N_\mathrm{train}$  (data vectors)", fontsize=AXES_FS)
    ax.set_ylabel(
        fr"95th pct  $\left| P_\mathrm{{emul}}(k) \,/\, P_\mathrm{{CAMB}}(k) - 1 \right|$"
        fr"  at  $k \approx 1 \, h/\mathrm{{Mpc}}$",
        fontsize=AXES_FS - 3,
    )
    ax.set_title(
        fr"Emulator accuracy vs. training set size  |  {z_label}",
        fontsize=AXES_FS - 2,
    )
    ax.set_xticks(n_dvs_list)
    ax.set_xticklabels([f"{nd // 1000}k" for nd in n_dvs_list], fontsize=TICK_FS)
    ax.tick_params(axis='y', labelsize=TICK_FS)
    ax.legend(fontsize=LEGEND_FS, loc="upper right")
    ax.grid(alpha=0.35)
    plt.tight_layout()

    fig_name = (f"ntrain_scaling_{args.prior_type}_{args.nl_type}"
                f"_iz{args.iz}_p95_k1_maxerr.pdf")
    fig_path = os.path.join(args.fig_dir, fig_name)
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"\n[INFO] Scaling plot saved to: {fig_path}")

    # --- Also save Syren scaling plot for reference ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for cosmo_type, cfg in COSMO_CONFIGS.items():
        p95_vals = np.array(results_syren[cosmo_type])
        valid    = ~np.isnan(p95_vals)
        x_plot   = np.array(n_dvs_list)[valid]
        y_plot   = p95_vals[valid]
        ax.plot(x_plot, y_plot,
                color=cfg["color"], marker=cfg["marker"],
                markersize=9, linewidth=2, label=cfg["label"],
                linestyle="--", zorder=3)
        for x, y in zip(x_plot, y_plot):
            ax.annotate(f"{y:.4f}", xy=(x, y), xytext=(0, 10),
                        textcoords="offset points", ha="center",
                        fontsize=ANNOT_FS, color=cfg["color"])

    ax.set_xlabel(r"$N_\mathrm{train}$  (data vectors)", fontsize=AXES_FS)
    ax.set_ylabel(
        fr"95th pct  $\left| P_\mathrm{{Syren}}(k) \,/\, P_\mathrm{{CAMB}}(k) - 1 \right|$"
        fr"  at  $k \approx 1 \, h/\mathrm{{Mpc}}$",
        fontsize=AXES_FS - 3,
    )
    ax.set_title(
        fr"EH approximation error vs. training set size  |  {z_label}",
        fontsize=AXES_FS - 2,
    )
    ax.set_xticks(n_dvs_list)
    ax.set_xticklabels([f"{nd // 1000}k" for nd in n_dvs_list], fontsize=TICK_FS)
    ax.tick_params(axis='y', labelsize=TICK_FS)
    ax.legend(fontsize=LEGEND_FS, loc="upper right")
    ax.grid(alpha=0.35)
    plt.tight_layout()

    fig_name_syr = (f"ntrain_scaling_syren_{args.prior_type}_{args.nl_type}"
                    f"_iz{args.iz}_p95_k1.pdf")
    plt.savefig(os.path.join(args.fig_dir, fig_name_syr), bbox_inches="tight")
    plt.close()
    print(f"[INFO] Syren scaling plot saved.")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()