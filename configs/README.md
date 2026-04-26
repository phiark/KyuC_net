# Configs

Store structured configuration files here.

- `model/`: architecture and hyperparameters
- `data/`: dataset and cohort construction
- `train/`: optimizer, scheduler, loss weights
- `eval/`: metrics and benchmark settings
- `analysis/`: plotting and report settings
- `study/`: multi-seed study wiring
- `protocol/`: manifest construction protocols

`plan_a_next_v0_2_*` configs are the data-semantics baseline. They use canonical `state_*` and `top1_completion_beta_*` metric names and reserve label-aware proposition fields for diagnostics.

Current active study line:

- `plan_a_next_v0_6c_*`: near-OOD CIFAR-100 class-holdout repair and workflow-integrity cleanup.

Historical configs remain for reproducibility:

- `plan_a_v0_3*`, `plan_a_next_v0_4*`, `plan_a_next_v0_5*`, `plan_a_next_v0_6*`, and `plan_a_next_v0_6b*` are retained but should not be used as the active evidence line unless the corresponding ADR/protocol is being reproduced.
