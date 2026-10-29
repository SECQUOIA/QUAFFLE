# Reproduce the figures

Use the original checkpoint/data source bindings and compatible project environment. Persistent local cache: `/home/bernalde/repos/QUAFFLE/results/hq_handoff_figures_20261028/predictions_v1`. Raw arrays and checkpoint weights are intentionally not published.

Rendering performs no model inference:

```bash
python misc/plot_hq_publication_figures.py \
  --cache-dir /path/to/the/complete/original/prediction-cache \
  --output-dir /path/to/new/empty/figure-output
```

If regenerating predictions is deliberately required, use the historical 60-epoch campaign and its retained original legacy root and native checkpoints:

```bash
python misc/collect_hq_validation_predictions.py \
  --work-dir /path/to/hq-training-controls/runs \
  --data-root /path/to/cleaned-DFC24 \
  --output-dir /path/to/new/empty/prediction-cache \
  --device cuda --gpu-memory-fraction YOUR_EXPLICIT_FRACTION \
  --budget-seconds YOUR_FINITE_SECONDS
```

The original validation counts must match exactly; mismatches stop collection. This command performs validation prediction of existing checkpoints and no training or test prediction. Keep failed/partial attempts. A layout change requires rendering only.

Heavy local commands were bounded by OS 6 GiB/no swap/two numerical threads, with a 12% CUDA allocator fraction during prediction. Those workstation limits are not cluster recommendations. Source/test hashes and exact commands appear in execution/*.binding.json and *.result.json.

To reproduce the launch package, use the operator guide and regenerate on Alan's execution checkout with his actual configuration. Do not submit the synthetic demonstration.
