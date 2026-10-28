# Full-size batch-one GPU probe dispositions

The original attempt failed before any model construction because the standalone helper passed `torch.device("cuda")` to a Torch memory-limit API requiring an explicit index. Its JSON, log and exact helper bytes are retained as `full_model_gpu_attempt1.*`.

The coordinator changed that helper call to `torch.device("cuda:0")`, kept the same full-size geometry, one-step budget, batch one, 12% CUDA cap and four-GiB/no-swap host scope, and launched a new output. All seven variants then passed finite forward/backward/Adam checks, all trainable core tensors received gradients with nonzero group movement, and the frozen core had no gradients and exactly zero movement. Individual unobservable/zero-gradient scalar angles remain permitted.

The shared initial backbone/projection hash agrees with the earlier CPU inference audit. Peak Torch allocated memory was at most 372101120 bytes (354.86 MiB), and peak reserved memory at most 446693376 bytes (426 MiB). These counters exclude display/other processes and CUDA allocations outside Torch. The probe uses random inputs, supplies no data accuracy or convergence evidence, and does not measure intended production batch eight or Anvil throughput.
