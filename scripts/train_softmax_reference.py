#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as tvf
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
RUNTIME_CACHE_ROOT = REPO_ROOT / ".cache" / "runtime"
RUNTIME_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))

from frcnet.data.plan_a import load_plan_a_source_datasets
from frcnet.models import SoftmaxReferenceModel
from frcnet.utils import resolve_pin_memory, resolve_runtime


class _IdentityClassDataset(Dataset):
    def __init__(self, source_dataset: object) -> None:
        self.source_dataset = source_dataset

    def __len__(self) -> int:
        return len(self.source_dataset)

    def __getitem__(self, index: int):
        image, label = self.source_dataset[index]
        if isinstance(image, torch.Tensor):
            image_tensor = image.detach().clone().float()
            if image_tensor.max() > 1:
                image_tensor = image_tensor / 255.0
        else:
            image_tensor = tvf.to_tensor(image)
        return image_tensor, int(label)


def _load_yaml_section(path: str | Path, section_name: str) -> dict[str, Any]:
    return dict(yaml.safe_load(Path(path).read_text(encoding="utf-8"))[section_name])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the same-backbone softmax CE reference model.")
    parser.add_argument("--protocol-config", default="configs/protocol/plan_a_next_v0_2_train.yaml")
    parser.add_argument("--model-config", default="configs/model/frcnet_resnet18_base.yaml")
    parser.add_argument("--reference-config", default="configs/reference/plan_a_next_v0_4_softmax_ce_reference.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="softmax_ce_reference")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol_config = _load_yaml_section(args.protocol_config, "protocol")
    model_config = _load_yaml_section(args.model_config, "model")
    reference_config = _load_yaml_section(args.reference_config, "reference_train")

    runtime_config = dict(reference_config.get("runtime", {}))
    runtime_spec = resolve_runtime(
        requested_backend=runtime_config.get("backend", "auto"),
        dtype=runtime_config.get("dtype", "float32"),
        amp_enabled=bool(runtime_config.get("amp_enabled", False)),
    )
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    record_dir = output_dir / "records"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)

    reference_protocol_config = dict(protocol_config)
    reference_protocol_config["datasets"] = {"cifar10": protocol_config["datasets"]["cifar10"]}
    source_datasets = load_plan_a_source_datasets(reference_protocol_config)
    train_dataset = _IdentityClassDataset(source_datasets["cifar10"])
    dataloader_config = dict(reference_config.get("dataloader", {}))
    dataloader = DataLoader(
        train_dataset,
        batch_size=int(dataloader_config.get("batch_size", 128)),
        shuffle=bool(dataloader_config.get("shuffle", True)),
        drop_last=bool(dataloader_config.get("drop_last", True)),
        num_workers=int(dataloader_config.get("num_workers", 0)),
        pin_memory=resolve_pin_memory(dataloader_config.get("pin_memory", "auto"), runtime_spec),
    )

    torch.manual_seed(int(reference_config.get("seed", 7)))
    model = SoftmaxReferenceModel(
        num_classes=int(model_config["num_classes"]),
        backbone_name=str(model_config.get("backbone", "resnet18")),
    ).to(runtime_spec.device)
    optimizer_config = dict(reference_config.get("optimizer", {}))
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(optimizer_config.get("lr", 0.01)),
        momentum=float(optimizer_config.get("momentum", 0.9)),
        weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
    )
    epoch_count = int(reference_config.get("epochs", 1))
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epoch_count + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0
        for image_batch, label_batch in dataloader:
            image_batch = image_batch.to(runtime_spec.device, dtype=runtime_spec.dtype)
            label_batch = label_batch.to(runtime_spec.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image_batch)
            loss = torch.nn.functional.cross_entropy(logits, label_batch)
            loss.backward()
            optimizer.step()
            batch_size = int(label_batch.shape[0])
            loss_sum += float(loss.item()) * batch_size
            correct += int((logits.argmax(dim=1) == label_batch).sum().item())
            count += batch_size
        history.append(
            {
                "epoch": epoch,
                "mean_loss": loss_sum / max(1, count),
                "top1_accuracy": correct / max(1, count),
                "sample_count": count,
            }
        )
        print(f"[softmax-reference] epoch={epoch} loss={history[-1]['mean_loss']:.6f}")

    checkpoint_path = checkpoint_dir / "checkpoint_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "run_id": args.run_id,
            "model_family": "softmax_ce_reference",
            "reference_score_name": "softmax_entropy",
            "epoch": epoch_count,
        },
        checkpoint_path,
    )
    summary_path = record_dir / "train_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "model_family": "softmax_ce_reference",
                "checkpoint_path": str(checkpoint_path),
                "history": history,
                "runtime": {
                    "requested_backend": runtime_spec.requested_backend,
                    "resolved_backend": runtime_spec.resolved_backend,
                    "device": str(runtime_spec.device),
                    "dtype": str(runtime_spec.dtype),
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(summary_path)
    print(checkpoint_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
