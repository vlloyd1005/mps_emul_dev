# """
# plot_delta_chi2_from_chains.py

# Reads the cobaya `evaluate` chain output files written by
# compute_delta_chi2.py (one per row, per model: DELTA_CHI2_CAMB_<i>.1.txt,
# DELTA_CHI2_EMUL_<i>.1.txt, DELTA_CHI2_EMUL2_<i>.1.txt, ...), pulls the chi2
# column out of each, matches each emulator variant against CAMB by row
# index, and for each variant: prints summary stats, saves a delta_chi2
# .npy, and plots |delta_chi2| as a log-x histogram. Also plots all variants
# overlaid on one histogram with different colors.

# To add a third variant later, just add its tag/pattern to EMUL_VARIANTS
# below -- everything else (stats, per-variant plot, overlay plot) picks it
# up automatically.

# USAGE:
#     python plot_delta_chi2_from_chains.py
# """

# import re
# import glob
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")  # no display on a compute node -- comment this out
#                         # (and uncomment plt.show() calls below) if running
#                         # somewhere with a display.
# import matplotlib.pyplot as plt

# # ---------------------------------------------------------------------------
# # Paths -- match compute_delta_chi2.py's output prefixes.
# # ---------------------------------------------------------------------------
# BASE = "/lustre/nvwulf/projects/MirandaGroup-nvwulf/victoria/cocoa/Cocoa/"
# CHAIN_DIR = BASE + "projects/roman_real/chains/emul_test/"
# CHAIN_DIR_NEW = BASE + "projects/roman_real/chains/emul_test_new/"

# CAMB_PATTERN = CHAIN_DIR + "DELTA_CHI2_CAMB_*.1.txt"
# IDX_RE_CAMB = re.compile(r"DELTA_CHI2_CAMB_(\d+)\.1\.txt$")

# # Each emulator variant to compare against CAMB: (tag, glob pattern, index regex).
# # Add more tuples here (e.g. a future "EMUL3") and the rest of the script
# # handles them automatically.
# EMUL_VARIANTS = [
#     ("EMUL", CHAIN_DIR + "DELTA_CHI2_EMUL_*.1.txt", re.compile(r"DELTA_CHI2_EMUL_(\d+)\.1\.txt$")),
#     # ("EMUL2", CHAIN_DIR + "DELTA_CHI2_EMUL2_*.1.txt", re.compile(r"DELTA_CHI2_EMUL2_(\d+)\.1\.txt$")),
#     # ("EMUL3", CHAIN_DIR + "DELTA_CHI2_EMUL3_*.1.txt", re.compile(r"DELTA_CHI2_EMUL3_(\d+)\.1\.txt$")),
#     # ("EMUL4", CHAIN_DIR + "DELTA_CHI2_EMUL4_*.1.txt", re.compile(r"DELTA_CHI2_EMUL4_(\d+)\.1\.txt$")),
#     # ("EMUL5", CHAIN_DIR + "DELTA_CHI2_EMUL5_*.1.txt", re.compile(r"DELTA_CHI2_EMUL5_(\d+)\.1\.txt$")),
#     # ("EMUL6", CHAIN_DIR + "DELTA_CHI2_EMUL6_*.1.txt", re.compile(r"DELTA_CHI2_EMUL6_(\d+)\.1\.txt$")),
#     # ("EMUL7", CHAIN_DIR + "DELTA_CHI2_EMUL7_*.1.txt", re.compile(r"DELTA_CHI2_EMUL7_(\d+)\.1\.txt$")),
#     # ("EMUL8", CHAIN_DIR + "DELTA_CHI2_EMUL8_*.1.txt", re.compile(r"DELTA_CHI2_EMUL8_(\d+)\.1\.txt$")),
#     # ("EMUL9", CHAIN_DIR + "DELTA_CHI2_EMUL9_*.1.txt", re.compile(r"DELTA_CHI2_EMUL9_(\d+)\.1\.txt$")),
#     # ("EMUL10", CHAIN_DIR + "DELTA_CHI2_EMUL10_*.1.txt", re.compile(r"DELTA_CHI2_EMUL10_(\d+)\.1\.txt$")),
#     # ("EMUL11", CHAIN_DIR + "DELTA_CHI2_EMUL11_*.1.txt", re.compile(r"DELTA_CHI2_EMUL11_(\d+)\.1\.txt$")),
#     # ("EMUL12", CHAIN_DIR + "DELTA_CHI2_EMUL12_*.1.txt", re.compile(r"DELTA_CHI2_EMUL12_(\d+)\.1\.txt$")),
#     # ("EMUL12", CHAIN_DIR + "DELTA_CHI2_EMUL12_*.1.txt", re.compile(r"DELTA_CHI2_EMUL12_(\d+)\.1\.txt$")),
#     # ("EMUL12", CHAIN_DIR_NEW + "DELTA_CHI2_EMUL12_*.1.txt", re.compile(r"DELTA_CHI2_EMUL12_(\d+)\.1\.txt$")),
#     ("EMUL", CHAIN_DIR_NEW + "DELTA_CHI2_EMUL_*.1.txt", re.compile(r"DELTA_CHI2_EMUL_(\d+)\.1\.txt$")),
# ]

# N_BINS = 30
# OVERLAY_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

# # ---------------------------------------------------------------------------
# # Omega_b / H0 triangle cut -- same construction as _apply_prior_cuts:
# # a line from (omegab_anchor, h0_max) to (omegab_max, h0_anchor); points
# # with H0 above that line (for their omegab) are cut.
# # ---------------------------------------------------------------------------
# OMEGAB_ANCHOR = 0.05
# H0_ANCHOR = 75.0
# OMEGAB_MAX = 0.072
# H0_MAX = 90.0

# # ---------------------------------------------------------------------------
# # Hypersphere cut -- mirrors the training-set construction: normalize each
# # cosmology into [-1, +1] using the (5%-stretched) test_priors_ml bounds,
# # then keep points with radius <= R in that 9D normalized space. w0wa is
# # reconstructed the same way (w0 + wa) since that's the actually-sampled
# # coordinate, not wa on its own.
# # ---------------------------------------------------------------------------
# TEST_PRIORS_ML = {
#     "As_1e9": (1.0, 3.5),
#     "ns": (0.8, 1.05),
#     "H0": (55, 88),
#     "Ob": (0.03, 0.07),
#     "Om": (0.2, 0.5),
#     "w0": (-2.0, -0.01),
#     "w0wa": (-4, -0.01),
#     "Tagn": (6.5, 8.0),
#     "mnu": (0.055, 0.6),
# }
# HYPERSPHERE_STRETCH = 0.05
# HYPERSPHERE_R = 1.3
# HYPERSPHERE_PARAM_ORDER = ["As_1e9", "ns", "H0", "Ob", "Om", "w0", "w0wa", "Tagn", "mnu"]


# def _compute_training_bounds():
#     bounds = {}
#     for name, (lo, hi) in TEST_PRIORS_ML.items():
#         width = hi - lo
#         bounds[name] = (lo - HYPERSPHERE_STRETCH * width, hi + HYPERSPHERE_STRETCH * width)
#     return bounds


# TRAINING_BOUNDS = _compute_training_bounds()


# def parse_chain_file(path):
#     """
#     Parses a cobaya `evaluate` (N=1) chain file: a single '#'-prefixed
#     header line of column names, followed by one data row of numbers.
#     Returns {column_name: value} or None if the file has no data row.
#     """
#     with open(path) as f:
#         lines = [line.rstrip("\n") for line in f if line.strip() != ""]
#     if not lines:
#         return None
#     header_line = lines[0]
#     if not header_line.startswith("#"):
#         raise ValueError(f"{path}: expected header line starting with '#'")
#     cols = header_line.lstrip("#").split()
#     data_lines = lines[1:]
#     if not data_lines:
#         return None
#     # N=1 evaluate runs should have exactly one data row -- take the first.
#     values = [float(x) for x in data_lines[0].split()]
#     if len(values) != len(cols):
#         raise ValueError(f"{path}: {len(cols)} columns but {len(values)} values")
#     return dict(zip(cols, values))


# def get_chi2(row_dict):
#     """
#     Prefers the likelihood-specific chi2 column (chi2__<likelihood_name>)
#     over the generic 'chi2' column -- same value when there's only one
#     likelihood, but this is robust if that ever changes.
#     """
#     for key in row_dict:
#         if key.startswith("chi2__"):
#             return row_dict[key]
#     return row_dict.get("chi2")


# def collect_rows(pattern, idx_re):
#     """
#     Returns {idx: full_row_dict} for every chain file matching pattern
#     that parses successfully and has a chi2 column. Keeping the full row
#     (not just chi2) lets the triangle cut below read omegab/H0 straight
#     out of the same files.
#     """
#     out = {}
#     for path in glob.glob(pattern):
#         m = idx_re.search(path)
#         if not m:
#             continue
#         idx = int(m.group(1))
#         try:
#             row = parse_chain_file(path)
#         except Exception as e:
#             print(f"Skipping {path}: failed to parse ({e})")
#             continue
#         if row is None:
#             print(f"Skipping {path}: no data row found (run may have failed)")
#             continue
#         if get_chi2(row) is None:
#             print(f"Skipping {path}: no chi2 column found")
#             continue
#         out[idx] = row
#     return out


# def triangle_cut_mask(idx_list, camb_rows):
#     """
#     Boolean mask (same order as idx_list): True = keep. Reads omegab/H0
#     from camb_rows for each index and applies the Omega_b/H0 triangle
#     cut: line from (OMEGAB_ANCHOR, H0_MAX) to (OMEGAB_MAX, H0_ANCHOR);
#     points with H0 above that line (for their omegab) are cut.
#     """
#     omegab_vals = np.array([camb_rows[i]["omegab"] for i in idx_list])
#     h0_vals = np.array([camb_rows[i]["H0"] for i in idx_list])

#     slope = (H0_ANCHOR - H0_MAX) / (OMEGAB_MAX - OMEGAB_ANCHOR)
#     h0_line = H0_MAX + slope * (omegab_vals - OMEGAB_ANCHOR)

#     return h0_vals <= h0_line


# def hypersphere_cut_mask(idx_list, camb_rows):
#     """
#     Boolean mask (same order as idx_list): True = keep. Reads the
#     cosmological params from camb_rows for each index, normalizes into
#     the stretched training box, and keeps points with radius <=
#     HYPERSPHERE_R.
#     """
#     pts = []
#     for i in idx_list:
#         row = camb_rows[i]
#         w0 = row["w"]
#         w0wa = row["w"] + row["wa"]  # reconstruct the actually-sampled coordinate
#         pts.append([
#             row["As_1e9"], row["ns"], row["H0"], row["omegab"], row["omegam"],
#             w0, w0wa, row["HMCode_logT_AGN"], row["mnu"],
#         ])
#     pts = np.array(pts)

#     lower = np.array([TRAINING_BOUNDS[name][0] for name in HYPERSPHERE_PARAM_ORDER])
#     upper = np.array([TRAINING_BOUNDS[name][1] for name in HYPERSPHERE_PARAM_ORDER])

#     normalized = 2 * (pts - lower) / (upper - lower) - 1
#     radius = np.sqrt(np.sum(normalized ** 2, axis=1))

#     return radius <= HYPERSPHERE_R


# def compute_delta(tag, camb_rows, variant_pattern, variant_idx_re):
#     """
#     Matches one emulator variant's chi2 values against camb_rows by row
#     index, applies the Omega_b/H0 triangle cut (reading omegab/H0 from
#     the CAMB row, since that's the physical cosmology both models were
#     evaluated at), prints summary stats, saves delta_chi2/indices .npy,
#     and returns (idx_arr, delta_chi2, abs_delta_chi2_nonzero) -- the last
#     for use in both the per-variant histogram and the overlay.
#     """
#     variant_rows = collect_rows(variant_pattern, variant_idx_re)
#     print(f"\n--- {tag} ---")
#     print(f"{tag} chain files parsed: {len(variant_rows)}")

#     common_idx = sorted(set(camb_rows) & set(variant_rows))
#     print(f"Matched indices (CAMB & {tag} both present): {len(common_idx)}")

#     missing_variant = sorted(set(camb_rows) - set(variant_rows))
#     missing_camb = sorted(set(variant_rows) - set(camb_rows))
#     if missing_variant:
#         print(f"Indices with CAMB output but no {tag} output ({len(missing_variant)}): {missing_variant}")
#     if missing_camb:
#         print(f"Indices with {tag} output but no CAMB output ({len(missing_camb)}): {missing_camb}")

#     if not common_idx:
#         print(f"No matched CAMB/{tag} pairs found -- skipping {tag}.")
#         return None

#     keep_mask = triangle_cut_mask(common_idx, camb_rows)
#     n_cut = (~keep_mask).sum()
#     print(f"  Omega_b/H0 triangle cut: removed {n_cut} cosmologies "
#           f"(line from Omega_b={OMEGAB_ANCHOR}, H0={H0_MAX} to Omega_b={OMEGAB_MAX}, H0={H0_ANCHOR})")
#     common_idx = [i for i, keep in zip(common_idx, keep_mask) if keep]
#     print(f"  -> {len(common_idx)} matched cosmologies remaining.")

#     if not common_idx:
#         print(f"All matched {tag} cosmologies removed by the triangle cut -- skipping {tag}.")
#         return None

#     keep_mask = hypersphere_cut_mask(common_idx, camb_rows)
#     n_cut = (~keep_mask).sum()
#     print(f"  Hypersphere cut (R={HYPERSPHERE_R}): removed {n_cut} additional cosmologies")
#     common_idx = [i for i, keep in zip(common_idx, keep_mask) if keep]
#     print(f"  -> {len(common_idx)} matched cosmologies remaining.")

#     if not common_idx:
#         print(f"All matched {tag} cosmologies removed by the hypersphere cut -- skipping {tag}.")
#         return None

#     idx_arr = np.array(common_idx)
#     delta_chi2 = np.array([get_chi2(camb_rows[i]) - get_chi2(variant_rows[i]) for i in common_idx])

#     np.save(CHAIN_DIR + f"delta_chi2_{tag.lower()}_from_chains.npy", delta_chi2)
#     np.save(CHAIN_DIR + f"delta_chi2_{tag.lower()}_indices.npy", idx_arr)

#     abs_delta_chi2 = np.abs(delta_chi2)
#     print(f"|delta_chi2| (camb - {tag.lower()}): n={len(abs_delta_chi2)}, "
#           f"mean={abs_delta_chi2.mean():.4g}, median={np.median(abs_delta_chi2):.4g}, "
#           f"min={abs_delta_chi2.min():.4g}, max={abs_delta_chi2.max():.4g}")

#     abs_delta_chi2_nonzero = abs_delta_chi2[abs_delta_chi2 > 0]

#     if abs_delta_chi2_nonzero.size == 0:
#         print(f"All |delta_chi2| values for {tag} are exactly 0 -- nothing to histogram.")
#         return None

#     return idx_arr, delta_chi2, abs_delta_chi2_nonzero


# def plot_single_histogram(tag, abs_delta_chi2):
#     bins = np.logspace(
#         np.log10(abs_delta_chi2.min()),
#         np.log10(abs_delta_chi2.max()),
#         N_BINS
#     )
#     plt.figure(figsize=(8, 5))
#     plt.hist(abs_delta_chi2, bins=bins)
#     plt.xscale('log')
#     plt.xlabel(r'$|\Delta\chi^2|$')
#     plt.ylabel('Number of samples')
#     plt.title(rf'Distribution of $|\Delta\chi^2|$ (CAMB vs {tag})')
#     plt.tight_layout()
#     out_png = CHAIN_DIR + f"delta_chi2_{tag.lower()}_histogram.png"
#     plt.savefig(out_png, dpi=150)
#     plt.close()
#     print(f"Saved {tag} histogram to {out_png}")
#     # plt.show()  # uncomment if running somewhere with a display


# def plot_overlay_histogram(results):
#     """results: list of (tag, abs_delta_chi2) for variants with data."""
#     if len(results) < 2:
#         print("Fewer than 2 variants with data -- skipping overlay plot.")
#         return

#     all_vals = np.concatenate([vals for _, vals in results])
#     bins = np.logspace(
#         np.log10(all_vals.min()),
#         np.log10(all_vals.max()),
#         N_BINS
#     )

#     plt.figure(figsize=(8, 5))
#     for (tag, vals), color in zip(results, OVERLAY_COLORS):
#         plt.hist(vals, bins=bins, alpha=0.5, label=f'CAMB vs {tag}', color=color)
#     plt.xscale('log')
#     plt.xlabel(r'$|\Delta\chi^2|$')
#     plt.ylabel('Number of samples')
#     plt.title(r'Distribution of $|\Delta\chi^2|$: emulator variants vs CAMB')
#     plt.legend()
#     plt.tight_layout()
#     out_png = CHAIN_DIR + "delta_chi2_overlay_histogram.png"
#     plt.savefig(out_png, dpi=150)
#     plt.close()
#     print(f"Saved overlay histogram to {out_png}")
#     # plt.show()  # uncomment if running somewhere with a display


# if __name__ == "__main__":

#     camb_rows = collect_rows(CAMB_PATTERN, IDX_RE_CAMB)
#     print(f"CAMB chain files parsed: {len(camb_rows)}")

#     if not camb_rows:
#         raise SystemExit("No CAMB chain files found -- check CHAIN_DIR / CAMB_PATTERN.")

#     overlay_data = []
#     for tag, pattern, idx_re in EMUL_VARIANTS:
#         result = compute_delta(tag, camb_rows, pattern, idx_re)
#         if result is None:
#             continue
#         idx_arr, delta_chi2, abs_delta_chi2 = result
#         plot_single_histogram(tag, abs_delta_chi2)
#         overlay_data.append((tag, abs_delta_chi2))

#     plot_overlay_histogram(overlay_data)

"""
plot_delta_chi2_from_chains.py

Reads the cobaya `evaluate` chain output files written by
compute_delta_chi2.py (one per row, per model: DELTA_CHI2_CAMB_<i>.1.txt,
DELTA_CHI2_EMUL_<i>.1.txt, DELTA_CHI2_EMUL2_<i>.1.txt, ...), pulls the chi2
column out of each, matches each emulator variant against CAMB by row
index, and for each variant: prints summary stats, saves a delta_chi2
.npy, and plots |delta_chi2| as a log-x histogram. Also plots all variants
overlaid on one histogram with different colors.

New in this version:
  - plot_triangle_delta_chi2(): a corner/triangle plot of the 9
    cosmological parameters, with every pairwise panel scatter-colored by
    log10(|delta_chi2|). Useful for spotting whether large errors
    cluster in a particular region of parameter space.
  - plot_delta_chi2_vs_params(): a 1xN strip of "delta_chi2 vs param_i"
    panels -- much easier to read for edge effects than the triangle
    panels alone.
  - plot_delta_chi2_vs_radius(): delta_chi2 vs the normalized
    hypersphere radius used for the training-set cut, to directly test
    whether error grows near the R=HYPERSPHERE_R boundary.

To add a third variant later, just add its tag/pattern to EMUL_VARIANTS
below -- everything else (stats, per-variant plots, overlay plot) picks
it up automatically.

USAGE:
    python plot_delta_chi2_from_chains.py
"""

import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display on a compute node -- comment this out
                        # (and uncomment plt.show() calls below) if running
                        # somewhere with a display.
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ---------------------------------------------------------------------------
# Paths -- match compute_delta_chi2.py's output prefixes.
# ---------------------------------------------------------------------------
BASE = "/lustre/nvwulf/projects/MirandaGroup-nvwulf/victoria/cocoa/Cocoa/"
CHAIN_DIR = BASE + "projects/roman_real/chains/emul_test/"
CHAIN_DIR_NEW = BASE + "projects/roman_real/chains/emul_test_new/"

CAMB_PATTERN = CHAIN_DIR + "DELTA_CHI2_CAMB_*.1.txt"
IDX_RE_CAMB = re.compile(r"DELTA_CHI2_CAMB_(\d+)\.1\.txt$")

# Each emulator variant to compare against CAMB: (tag, glob pattern, index regex).
# Add more tuples here (e.g. a future "EMUL3") and the rest of the script
# handles them automatically.
EMUL_VARIANTS = [
    ("EMUL", CHAIN_DIR + "DELTA_CHI2_EMUL_*.1.txt", re.compile(r"DELTA_CHI2_EMUL_(\d+)\.1\.txt$")),
    # ("EMUL2", CHAIN_DIR + "DELTA_CHI2_EMUL2_*.1.txt", re.compile(r"DELTA_CHI2_EMUL2_(\d+)\.1\.txt$")),
    # ("EMUL3", CHAIN_DIR + "DELTA_CHI2_EMUL3_*.1.txt", re.compile(r"DELTA_CHI2_EMUL3_(\d+)\.1\.txt$")),
    # ("EMUL4", CHAIN_DIR + "DELTA_CHI2_EMUL4_*.1.txt", re.compile(r"DELTA_CHI2_EMUL4_(\d+)\.1\.txt$")),
    # ("EMUL5", CHAIN_DIR + "DELTA_CHI2_EMUL5_*.1.txt", re.compile(r"DELTA_CHI2_EMUL5_(\d+)\.1\.txt$")),
    # ("EMUL6", CHAIN_DIR + "DELTA_CHI2_EMUL6_*.1.txt", re.compile(r"DELTA_CHI2_EMUL6_(\d+)\.1\.txt$")),
    # ("EMUL7", CHAIN_DIR + "DELTA_CHI2_EMUL7_*.1.txt", re.compile(r"DELTA_CHI2_EMUL7_(\d+)\.1\.txt$")),
    # ("EMUL8", CHAIN_DIR + "DELTA_CHI2_EMUL8_*.1.txt", re.compile(r"DELTA_CHI2_EMUL8_(\d+)\.1\.txt$")),
    # ("EMUL9", CHAIN_DIR + "DELTA_CHI2_EMUL9_*.1.txt", re.compile(r"DELTA_CHI2_EMUL9_(\d+)\.1\.txt$")),
    # ("EMUL10", CHAIN_DIR + "DELTA_CHI2_EMUL10_*.1.txt", re.compile(r"DELTA_CHI2_EMUL10_(\d+)\.1\.txt$")),
    # ("EMUL11", CHAIN_DIR + "DELTA_CHI2_EMUL11_*.1.txt", re.compile(r"DELTA_CHI2_EMUL11_(\d+)\.1\.txt$")),
    # ("EMUL12", CHAIN_DIR + "DELTA_CHI2_EMUL12_*.1.txt", re.compile(r"DELTA_CHI2_EMUL12_(\d+)\.1\.txt$")),
    # ("EMUL12", CHAIN_DIR_NEW + "DELTA_CHI2_EMUL12_*.1.txt", re.compile(r"DELTA_CHI2_EMUL12_(\d+)\.1\.txt$")),
    ("EMUL", CHAIN_DIR_NEW + "DELTA_CHI2_EMUL_*.1.txt", re.compile(r"DELTA_CHI2_EMUL_(\d+)\.1\.txt$")),
    ("EMUL2", CHAIN_DIR_NEW + "DELTA_CHI2_EMUL2_*.1.txt", re.compile(r"DELTA_CHI2_EMUL2_(\d+)\.1\.txt$")),
    ("EMUL3", CHAIN_DIR_NEW + "DELTA_CHI2_EMUL3_*.1.txt", re.compile(r"DELTA_CHI2_EMUL3_(\d+)\.1\.txt$")),
    ("EMUL4", CHAIN_DIR_NEW + "DELTA_CHI2_EMUL4_*.1.txt", re.compile(r"DELTA_CHI2_EMUL4_(\d+)\.1\.txt$")),
    ("EMUL5", CHAIN_DIR_NEW + "DELTA_CHI2_EMUL5_*.1.txt", re.compile(r"DELTA_CHI2_EMUL5_(\d+)\.1\.txt$"))
]

N_BINS = 30
OVERLAY_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

# ---------------------------------------------------------------------------
# Omega_b / H0 triangle cut -- same construction as _apply_prior_cuts:
# a line from (omegab_anchor, h0_max) to (omegab_max, h0_anchor); points
# with H0 above that line (for their omegab) are cut.
# ---------------------------------------------------------------------------
OMEGAB_ANCHOR = 0.05
H0_ANCHOR = 75.0
OMEGAB_MAX = 0.072
H0_MAX = 90.0

OMEGAM_ANCHOR = 0.38
AS_ANCHOR = 3
OMEGAM_MAX = 0.5
AS_MAX = 3.5

# ---------------------------------------------------------------------------
# Hypersphere cut -- mirrors the training-set construction: normalize each
# cosmology into [-1, +1] using the (5%-stretched) test_priors_ml bounds,
# then keep points with radius <= R in that 9D normalized space. w0wa is
# reconstructed the same way (w0 + wa) since that's the actually-sampled
# coordinate, not wa on its own.
# ---------------------------------------------------------------------------
TEST_PRIORS_ML = {
    "As_1e9": (1.0, 3.5),
    "ns": (0.8, 1.05),
    "H0": (55, 88),
    "Ob": (0.03, 0.07),
    "Om": (0.2, 0.5),
    "w0": (-2.0, -0.01),
    "w0wa": (-4, -0.01),
    "Tagn": (6.5, 8.0),
    "mnu": (0.055, 0.6),
}
HYPERSPHERE_STRETCH = 0.05
HYPERSPHERE_R = 1.3
HYPERSPHERE_PARAM_ORDER = ["As_1e9", "ns", "H0", "Ob", "Om", "w0", "w0wa", "Tagn", "mnu"]

# Nicer axis labels for plotting (LaTeX-ish, matches params_latex convention
# used in the training utils module).
PARAM_LABELS = {
    "As_1e9": r"$10^9 A_s$",
    "ns": r"$n_s$",
    "H0": r"$H_0$",
    "Ob": r"$\Omega_b$",
    "Om": r"$\Omega_m$",
    "w0": r"$w_0$",
    "w0wa": r"$w_0+w_a$",
    "Tagn": r"$\log T_{\rm AGN}$",
    "mnu": r"$m_\nu$",
}


def _compute_training_bounds():
    bounds = {}
    for name, (lo, hi) in TEST_PRIORS_ML.items():
        width = hi - lo
        bounds[name] = (lo - HYPERSPHERE_STRETCH * width, hi + HYPERSPHERE_STRETCH * width)
    return bounds


TRAINING_BOUNDS = _compute_training_bounds()


def parse_chain_file(path):
    """
    Parses a cobaya `evaluate` (N=1) chain file: a single '#'-prefixed
    header line of column names, followed by one data row of numbers.
    Returns {column_name: value} or None if the file has no data row.
    """
    with open(path) as f:
        lines = [line.rstrip("\n") for line in f if line.strip() != ""]
    if not lines:
        return None
    header_line = lines[0]
    if not header_line.startswith("#"):
        raise ValueError(f"{path}: expected header line starting with '#'")
    cols = header_line.lstrip("#").split()
    data_lines = lines[1:]
    if not data_lines:
        return None
    # N=1 evaluate runs should have exactly one data row -- take the first.
    values = [float(x) for x in data_lines[0].split()]
    if len(values) != len(cols):
        raise ValueError(f"{path}: {len(cols)} columns but {len(values)} values")
    return dict(zip(cols, values))


def get_chi2(row_dict):
    """
    Prefers the likelihood-specific chi2 column (chi2__<likelihood_name>)
    over the generic 'chi2' column -- same value when there's only one
    likelihood, but this is robust if that ever changes.
    """
    for key in row_dict:
        if key.startswith("chi2__"):
            return row_dict[key]
    return row_dict.get("chi2")


def collect_rows(pattern, idx_re):
    """
    Returns {idx: full_row_dict} for every chain file matching pattern
    that parses successfully and has a chi2 column. Keeping the full row
    (not just chi2) lets the triangle cut below (and the parameter plots)
    read omegab/H0/etc straight out of the same files.
    """
    out = {}
    for path in glob.glob(pattern):
        m = idx_re.search(path)
        if not m:
            continue
        idx = int(m.group(1))
        try:
            row = parse_chain_file(path)
        except Exception as e:
            print(f"Skipping {path}: failed to parse ({e})")
            continue
        if row is None:
            print(f"Skipping {path}: no data row found (run may have failed)")
            continue
        if get_chi2(row) is None:
            print(f"Skipping {path}: no chi2 column found")
            continue
        out[idx] = row
    return out


def triangle_cut_mask(idx_list, camb_rows):
    """
    Boolean mask (same order as idx_list): True = keep. Reads omegab/H0
    from camb_rows for each index and applies the Omega_b/H0 triangle
    cut: line from (OMEGAB_ANCHOR, H0_MAX) to (OMEGAB_MAX, H0_ANCHOR);
    points with H0 above that line (for their omegab) are cut.
    """
    omegab_vals = np.array([camb_rows[i]["omegab"] for i in idx_list])
    h0_vals = np.array([camb_rows[i]["H0"] for i in idx_list])

    slope = (H0_ANCHOR - H0_MAX) / (OMEGAB_MAX - OMEGAB_ANCHOR)
    h0_line = H0_MAX + slope * (omegab_vals - OMEGAB_ANCHOR)

    return h0_vals <= h0_line

def triangle_cut_mask_as(idx_list, camb_rows):
    """
    Boolean mask (same order as idx_list): True = keep. Reads omegab/H0
    from camb_rows for each index and applies the Omega_b/H0 triangle
    cut: line from (OMEGAB_ANCHOR, H0_MAX) to (OMEGAB_MAX, H0_ANCHOR);
    points with H0 above that line (for their omegab) are cut.
    """
    omegam_vals = np.array([camb_rows[i]["omegam"] for i in idx_list])
    as_vals = np.array([camb_rows[i]["As"] for i in idx_list])

    slope = (AS_ANCHOR - AS_MAX) / (OMEGAM_MAX - OMEGAM_ANCHOR)
    as_line = AS_MAX + slope * (omegam_vals - OMEGAM_ANCHOR)

    return as_vals <= as_line


def get_params_array(idx_list, camb_rows, param_order=HYPERSPHERE_PARAM_ORDER):
    """
    Returns an (N, len(param_order)) array of raw (un-normalized)
    parameter values for idx_list, pulled from camb_rows, in
    param_order. w0wa is reconstructed as row['w'] + row['wa'] (the
    actually-sampled combination), matching hypersphere_cut_mask.
    """
    pts = []
    for i in idx_list:
        row = camb_rows[i]
        w0 = row["w"]
        w0wa = row["w"] + row["wa"]
        lookup = {
            "As_1e9": row["As_1e9"],
            "ns": row["ns"],
            "H0": row["H0"],
            "Ob": row["omegab"],
            "Om": row["omegam"],
            "w0": w0,
            "w0wa": w0wa,
            "Tagn": row["HMCode_logT_AGN"],
            "mnu": row["mnu"],
        }
        pts.append([lookup[name] for name in param_order])
    return np.array(pts)


def get_normalized_params_array(idx_list, camb_rows, param_order=HYPERSPHERE_PARAM_ORDER):
    """
    Same as get_params_array but normalized into [-1, 1] using
    TRAINING_BOUNDS -- i.e. the same coordinates the hypersphere radius
    cut is computed in. Useful for the radius diagnostic and for
    per-parameter edge-proximity checks.
    """
    pts = get_params_array(idx_list, camb_rows, param_order=param_order)
    lower = np.array([TRAINING_BOUNDS[name][0] for name in param_order])
    upper = np.array([TRAINING_BOUNDS[name][1] for name in param_order])
    return 2 * (pts - lower) / (upper - lower) - 1


def hypersphere_cut_mask(idx_list, camb_rows):
    """
    Boolean mask (same order as idx_list): True = keep. Reads the
    cosmological params from camb_rows for each index, normalizes into
    the stretched training box, and keeps points with radius <=
    HYPERSPHERE_R.
    """
    normalized = get_normalized_params_array(idx_list, camb_rows)
    radius = np.sqrt(np.sum(normalized ** 2, axis=1))
    return radius <= HYPERSPHERE_R


def compute_delta(tag, camb_rows, variant_pattern, variant_idx_re):
    """
    Matches one emulator variant's chi2 values against camb_rows by row
    index, applies the Omega_b/H0 triangle cut and hypersphere cut
    (reading params from the CAMB row, since that's the physical
    cosmology both models were evaluated at), prints summary stats,
    saves delta_chi2/indices .npy, and returns
    (idx_arr, delta_chi2, abs_delta_chi2_nonzero) -- idx_arr and
    delta_chi2 are full-length and index-aligned (safe to zip with
    get_params_array(idx_arr, camb_rows)); abs_delta_chi2_nonzero is
    filtered (zeros dropped) and used only for the histograms.
    """
    variant_rows = collect_rows(variant_pattern, variant_idx_re)
    print(f"\n--- {tag} ---")
    print(f"{tag} chain files parsed: {len(variant_rows)}")

    common_idx = sorted(set(camb_rows) & set(variant_rows))
    print(f"Matched indices (CAMB & {tag} both present): {len(common_idx)}")

    missing_variant = sorted(set(camb_rows) - set(variant_rows))
    missing_camb = sorted(set(variant_rows) - set(camb_rows))
    if missing_variant:
        print(f"Indices with CAMB output but no {tag} output ({len(missing_variant)}): {missing_variant}")
    if missing_camb:
        print(f"Indices with {tag} output but no CAMB output ({len(missing_camb)}): {missing_camb}")

    if not common_idx:
        print(f"No matched CAMB/{tag} pairs found -- skipping {tag}.")
        return None

    keep_mask = triangle_cut_mask(common_idx, camb_rows)
    n_cut = (~keep_mask).sum()
    print(f"  Omega_b/H0 triangle cut: removed {n_cut} cosmologies "
          f"(line from Omega_b={OMEGAB_ANCHOR}, H0={H0_MAX} to Omega_b={OMEGAB_MAX}, H0={H0_ANCHOR})")
    common_idx = [i for i, keep in zip(common_idx, keep_mask) if keep]
    print(f"  -> {len(common_idx)} matched cosmologies remaining.")

    keep_mask = triangle_cut_mask_as(common_idx, camb_rows)
    n_cut = (~keep_mask).sum()
    print(f"  Omega_m/As triangle cut: removed {n_cut} cosmologies "
          f"(line from Omega_m={OMEGAM_ANCHOR}, H0={AS_MAX} to Omega_m={OMEGAM_MAX}, As={AS_ANCHOR})")
    common_idx = [i for i, keep in zip(common_idx, keep_mask) if keep]
    print(f"  -> {len(common_idx)} matched cosmologies remaining.")

    if not common_idx:
        print(f"All matched {tag} cosmologies removed by the triangle cut -- skipping {tag}.")
        return None

    keep_mask = hypersphere_cut_mask(common_idx, camb_rows)
    n_cut = (~keep_mask).sum()
    print(f"  Hypersphere cut (R={HYPERSPHERE_R}): removed {n_cut} additional cosmologies")
    common_idx = [i for i, keep in zip(common_idx, keep_mask) if keep]
    print(f"  -> {len(common_idx)} matched cosmologies remaining.")

    if not common_idx:
        print(f"All matched {tag} cosmologies removed by the hypersphere cut -- skipping {tag}.")
        return None

    idx_arr = np.array(common_idx)
    delta_chi2 = np.array([get_chi2(camb_rows[i]) - get_chi2(variant_rows[i]) for i in common_idx])

    np.save(CHAIN_DIR + f"delta_chi2_{tag.lower()}_from_chains.npy", delta_chi2)
    np.save(CHAIN_DIR + f"delta_chi2_{tag.lower()}_indices.npy", idx_arr)

    abs_delta_chi2 = np.abs(delta_chi2)
    print(f"|delta_chi2| (camb - {tag.lower()}): n={len(abs_delta_chi2)}, "
          f"mean={abs_delta_chi2.mean():.4g}, median={np.median(abs_delta_chi2):.4g}, "
          f"min={abs_delta_chi2.min():.4g}, max={abs_delta_chi2.max():.4g}")

    abs_delta_chi2_nonzero = abs_delta_chi2[abs_delta_chi2 > 0]

    if abs_delta_chi2_nonzero.size == 0:
        print(f"All |delta_chi2| values for {tag} are exactly 0 -- nothing to histogram.")
        return None

    return idx_arr, delta_chi2, abs_delta_chi2_nonzero


def plot_single_histogram(tag, abs_delta_chi2):
    bins = np.logspace(
        np.log10(abs_delta_chi2.min()),
        np.log10(abs_delta_chi2.max()),
        N_BINS
    )
    plt.figure(figsize=(8, 5))
    plt.hist(abs_delta_chi2, bins=bins)
    plt.xscale('log')
    plt.xlabel(r'$|\Delta\chi^2|$')
    plt.ylabel('Number of samples')
    plt.title(rf'Distribution of $|\Delta\chi^2|$ (CAMB vs {tag})')
    plt.tight_layout()
    out_png = CHAIN_DIR + f"delta_chi2_{tag.lower()}_histogram.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved {tag} histogram to {out_png}")
    # plt.show()  # uncomment if running somewhere with a display


def plot_overlay_histogram(results):
    """results: list of (tag, abs_delta_chi2) for variants with data."""
    if len(results) < 2:
        print("Fewer than 2 variants with data -- skipping overlay plot.")
        return

    all_vals = np.concatenate([vals for _, vals in results])
    bins = np.logspace(
        np.log10(all_vals.min()),
        np.log10(all_vals.max()),
        N_BINS
    )

    plt.figure(figsize=(8, 5))
    for (tag, vals), color in zip(results, OVERLAY_COLORS):
        plt.hist(vals, bins=bins, alpha=0.5, label=f'CAMB vs {tag}', color=color)
    plt.xscale('log')
    plt.xlabel(r'$|\Delta\chi^2|$')
    plt.ylabel('Number of samples')
    plt.title(r'Distribution of $|\Delta\chi^2|$: emulator variants vs CAMB')
    plt.legend()
    plt.tight_layout()
    out_png = CHAIN_DIR + "delta_chi2_overlay_histogram.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved overlay histogram to {out_png}")
    # plt.show()  # uncomment if running somewhere with a display


def _color_scale(abs_delta_chi2, floor=1e-3):
    """
    log10(|delta_chi2| + floor) color values, plus a Normalize object
    clipped to the 2nd/98th percentile so a handful of extreme outliers
    don't wash out the color scale for everything else. floor guards
    log10(0) for exact-zero delta_chi2 entries.
    """
    log_vals = np.log10(abs_delta_chi2 + floor)
    vmin, vmax = np.percentile(log_vals, [2, 98])
    if vmin == vmax:  # degenerate (e.g. all identical) -- fall back to full range
        vmin, vmax = log_vals.min(), log_vals.max()
    return log_vals, Normalize(vmin=vmin, vmax=vmax)


def plot_triangle_delta_chi2(tag, idx_arr, delta_chi2, camb_rows,
                              param_order=HYPERSPHERE_PARAM_ORDER, point_size=6):
    """
    Corner/triangle plot of the cosmological parameters (lower-triangle
    pairwise scatter, diagonal = 1D histograms), with every scatter
    panel colored by log10(|delta_chi2|). Uses a shared color scale
    across all panels (with a colorbar) so panels are comparable.
    """
    params = get_params_array(idx_arr, camb_rows, param_order=param_order)
    abs_dchi2 = np.abs(delta_chi2)
    color_vals, norm = _color_scale(abs_dchi2)
    cmap = plt.get_cmap("viridis")

    n = len(param_order)
    fig, axes = plt.subplots(n, n, figsize=(2.1 * n, 2.1 * n))

    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            if col > row:
                ax.axis("off")
                continue
            if row == col:
                ax.hist(params[:, row], bins=25, color="0.6")
                ax.set_yticks([])
            else:
                sc = ax.scatter(
                    params[:, col], params[:, row],
                    c=color_vals, cmap=cmap, norm=norm,
                    s=point_size, linewidths=0,
                )
            if row < n - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(PARAM_LABELS.get(param_order[col], param_order[col]))
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            if col > 0 or row == 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(PARAM_LABELS.get(param_order[row], param_order[row]))

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label(r"$\log_{10}(|\Delta\chi^2| + 10^{-3})$")

    fig.suptitle(rf"Parameter triangle plot colored by $|\Delta\chi^2|$ (CAMB vs {tag})", y=0.995)
    out_png = CHAIN_DIR + f"delta_chi2_{tag.lower()}_triangle.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {tag} triangle plot to {out_png}")
    # plt.show()  # uncomment if running somewhere with a display


def plot_delta_chi2_vs_params(tag, idx_arr, delta_chi2, camb_rows,
                               param_order=HYPERSPHERE_PARAM_ORDER):
    """
    1xN strip of delta_chi2 (signed) vs each raw parameter value. Much
    easier to spot prior-edge blowups here than in the triangle panels:
    look for points fanning out / growing in magnitude near the left or
    right edge of any panel's x-range.
    """
    params = get_params_array(idx_arr, camb_rows, param_order=param_order)
    n = len(param_order)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.2), sharey=True)

    for i, ax in enumerate(axes):
        ax.scatter(params[:, i], delta_chi2, s=6, alpha=0.5, color="#1f77b4")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel(PARAM_LABELS.get(param_order[i], param_order[i]))
        ax.set_yscale("symlog", linthresh=1.0)

    axes[0].set_ylabel(r"$\Delta\chi^2$ (CAMB $-$ " + tag + ")")
    fig.suptitle(rf"$\Delta\chi^2$ vs. each parameter (CAMB vs {tag})")
    fig.tight_layout()
    out_png = CHAIN_DIR + f"delta_chi2_{tag.lower()}_vs_params.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved {tag} delta_chi2-vs-params strip plot to {out_png}")
    # plt.show()  # uncomment if running somewhere with a display


def plot_delta_chi2_vs_radius(tag, idx_arr, delta_chi2, camb_rows):
    """
    delta_chi2 (signed) vs. the normalized hypersphere radius used for
    the training-set cut (HYPERSPHERE_R). If error grows as points
    approach R, that's direct evidence the emulator is degrading near
    the edge of its training coverage rather than failing randomly.
    """
    normalized = get_normalized_params_array(idx_arr, camb_rows)
    radius = np.sqrt(np.sum(normalized ** 2, axis=1))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(radius, delta_chi2, s=8, alpha=0.5, color="#d62728")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(HYPERSPHERE_R, color="0.4", ls="--", lw=1, label=f"training cut R={HYPERSPHERE_R}")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("Normalized hypersphere radius")
    ax.set_ylabel(r"$\Delta\chi^2$ (CAMB $-$ " + tag + ")")
    ax.set_title(rf"$\Delta\chi^2$ vs. training-box radius (CAMB vs {tag})")
    ax.legend()
    fig.tight_layout()
    out_png = CHAIN_DIR + f"delta_chi2_{tag.lower()}_vs_radius.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved {tag} delta_chi2-vs-radius plot to {out_png}")
    # plt.show()  # uncomment if running somewhere with a display


if __name__ == "__main__":

    camb_rows = collect_rows(CAMB_PATTERN, IDX_RE_CAMB)
    print(f"CAMB chain files parsed: {len(camb_rows)}")

    if not camb_rows:
        raise SystemExit("No CAMB chain files found -- check CHAIN_DIR / CAMB_PATTERN.")

    overlay_data = []
    for tag, pattern, idx_re in EMUL_VARIANTS:
        result = compute_delta(tag, camb_rows, pattern, idx_re)
        if result is None:
            continue
        idx_arr, delta_chi2, abs_delta_chi2 = result

        plot_single_histogram(tag, abs_delta_chi2)
        overlay_data.append((tag, abs_delta_chi2))

        # Parameter-space diagnostics -- use the full (index-aligned)
        # idx_arr/delta_chi2, not the zero-filtered abs_delta_chi2.
        plot_triangle_delta_chi2(tag, idx_arr, delta_chi2, camb_rows)
        plot_delta_chi2_vs_params(tag, idx_arr, delta_chi2, camb_rows)
        plot_delta_chi2_vs_radius(tag, idx_arr, delta_chi2, camb_rows)

    plot_overlay_histogram(overlay_data)