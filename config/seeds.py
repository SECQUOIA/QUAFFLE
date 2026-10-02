"""Single pre-declared seed list for every QUAFFLE experiment and analysis.

Every SLURM generator, table builder, forest plot and analyze_results.py
imports SEEDS from here. Do not filter it per method after looking at the
results. If a run for one of these seeds is missing or crashed, the analysis
scripts report it as missing (with n and the seed ids) instead of silently
switching to a smaller pool.

Verified on 2026-10-02 by listing results/*_seed<N>/ in the main checkout:
exactly these 10 seeds have run directories. Seed 42 has 3 quantum runs
without a Test block (centralized combined, fed random sar, fed dirichlet
sar), which the analysis reports as missing.
"""

SEEDS = [7, 17, 31, 42, 77, 123, 256, 314, 512, 999]

# Number of seeds the analysis expects per (method, dataset) cell.
N_SEEDS = len(SEEDS)
