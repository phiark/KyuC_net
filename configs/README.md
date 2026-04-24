# Configs

Store structured configuration files here.

- `model/`: architecture and hyperparameters
- `data/`: dataset and cohort construction
- `train/`: optimizer, scheduler, loss weights
- `eval/`: metrics and benchmark settings
- `analysis/`: plotting and report settings
- `study/`: multi-seed study wiring
- `protocol/`: manifest construction protocols

`plan_a_next_v0_2_*` configs are the current data-semantics baseline. They use canonical `state_*` and `top1_completion_beta_*` metric names and reserve label-aware proposition fields for diagnostics.
