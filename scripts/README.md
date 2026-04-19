# Scripts

This directory is for repeatable entrypoints only.

Rules:

- each script should map to a documented workflow
- scripts should not hide configuration in code
- scripts should emit run identifiers and output paths

Current protocol chain:

- `prepare_plan_a_data.py`: verify or download CIFAR10 / SVHN assets required by the train and analysis protocols
- `build_plan_a_manifest.py`: build a manifest snapshot from the Plan A protocol config
- `train_plan_a.py`: build or reuse a training manifest, train FRCNet, and emit checkpoints plus train records
- `run_plan_a_inference.py`: load the manifest, export sample analysis records, and export the top-1 proposition view
- `generate_plan_a_artifacts.py`: generate paper-facing figures, tables, artifact index, and experiment record
- `run_plan_a_experiment.py`: run dataset preflight, training, analysis inference, and artifact generation as one repeatable bundle
