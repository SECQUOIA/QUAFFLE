"""Single pre-declared seed list for every QUAFFLE experiment and analysis.

Every SLURM generator, table builder, forest plot and analyze_results.py
imports SEEDS from here. Do not filter it per method after looking at the
results. If a run for one of these seeds is missing or crashed, the analysis
scripts report it as missing (with n and the seed ids) instead of silently
switching to a smaller pool.

SEEDS is the full list of seeds used for every method, mode and dataset.
"""

SEEDS = [7, 17, 31, 42, 77, 123, 256, 314, 512, 999]

# Number of seeds the analysis expects per (method, dataset) cell.
N_SEEDS = len(SEEDS)
