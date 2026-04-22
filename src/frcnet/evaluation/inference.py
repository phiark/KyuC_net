from __future__ import annotations

from typing import Iterable

import torch
from torch.utils.data import DataLoader

from frcnet.data import BatchInput
from frcnet.evaluation.records import DEFAULT_MODEL_FAMILY, SampleAnalysisRecord, Top1PropositionRecord
from frcnet.models import FRCNetModel, ModelOutput
from frcnet.utils import (
    completion_score,
    content_entropy,
    move_batch_to_device,
    resolution_entropy,
    resolution_weighted_content_entropy,
    ternary_entropy_from_masses,
)


def _build_proposition_target_mask(batch_input: BatchInput, num_classes: int, device: torch.device) -> torch.Tensor:
    target_mask = torch.zeros((batch_input.batch_size, num_classes), dtype=torch.bool, device=device)
    for index, cohort_name in enumerate(batch_input.cohort_name):
        if cohort_name in {"easy_id", "hard_id"}:
            target_mask[index, int(batch_input.class_label[index].item())] = True
        elif cohort_name == "ambiguous_id":
            if batch_input.candidate_class_mask is None:
                raise ValueError("candidate_class_mask is required for ambiguous_id samples.")
            target_mask[index] = batch_input.candidate_class_mask[index].to(device=device)
    return target_mask


def _proposition_target_type(cohort_name: str) -> str:
    if cohort_name in {"easy_id", "hard_id"}:
        return "single_label"
    if cohort_name == "ambiguous_id":
        return "candidate_set"
    return "empty_set"


def build_sample_analysis_records(
    model_output: ModelOutput,
    batch_input: BatchInput,
    run_id: str,
    protocol_id: str,
    *,
    model_family: str = DEFAULT_MODEL_FAMILY,
) -> list[SampleAnalysisRecord]:
    predicted_class_index = torch.argmax(model_output.class_mass, dim=1)
    auxiliary_top1_content_probability = model_output.content_distribution.gather(
        1, predicted_class_index.unsqueeze(1)
    ).squeeze(1)
    entropy = content_entropy(model_output.content_distribution)
    weighted_entropy = resolution_weighted_content_entropy(model_output.resolution_ratio, entropy)
    resolution_entropy_value = resolution_entropy(model_output.resolution_ratio)
    score_beta_0_1 = completion_score(model_output.class_mass, model_output.unknown_mass, beta=0.1)
    score_beta_0_25 = completion_score(model_output.class_mass, model_output.unknown_mass, beta=0.25)
    score_beta_0_5 = completion_score(model_output.class_mass, model_output.unknown_mass, beta=0.5)
    score_beta_0_75 = completion_score(model_output.class_mass, model_output.unknown_mass, beta=0.75)
    top1_class_mass = model_output.class_mass.gather(1, predicted_class_index.unsqueeze(1)).squeeze(1)
    proposition_target_mask = _build_proposition_target_mask(
        batch_input,
        model_output.num_classes,
        model_output.class_mass.device,
    ).to(dtype=model_output.class_mass.dtype)
    proposition_truth_mass = (model_output.class_mass * proposition_target_mask).sum(dim=1)
    proposition_false_mass = (model_output.class_mass.sum(dim=1) - proposition_truth_mass).clamp_min(0.0)
    proposition_unknown_mass = model_output.unknown_mass
    proposition_truth_ratio = torch.zeros_like(proposition_truth_mass)
    resolved_mass = proposition_truth_mass + proposition_false_mass
    resolved_mask = resolved_mass > torch.finfo(resolved_mass.dtype).eps
    proposition_truth_ratio[resolved_mask] = proposition_truth_mass[resolved_mask] / resolved_mass[resolved_mask]
    ternary_entropy_value = ternary_entropy_from_masses(
        proposition_truth_mass,
        proposition_false_mass,
        proposition_unknown_mass,
    )

    record_list: list[SampleAnalysisRecord] = []
    candidate_class_mask = batch_input.candidate_class_mask
    for index in range(batch_input.batch_size):
        candidate_indices: tuple[int, ...] = ()
        if candidate_class_mask is not None:
            candidate_indices = tuple(torch.nonzero(candidate_class_mask[index], as_tuple=False).squeeze(1).tolist())
        source_class_label = None
        if batch_input.source_class_label is not None:
            source_class_label = batch_input.source_class_label[index]

        record_list.append(
            SampleAnalysisRecord(
                model_family=model_family,
                run_id=run_id,
                protocol_id=protocol_id,
                sample_id=batch_input.sample_id[index],
                split_name=batch_input.split_name[index],
                cohort_name=batch_input.cohort_name[index],
                source_dataset_name=batch_input.source_dataset_name[index],
                source_class_label=source_class_label,
                class_label=int(batch_input.class_label[index].item()),
                predicted_class_index=int(predicted_class_index[index].item()),
                resolution_ratio=float(model_output.resolution_ratio[index].item()),
                unknown_mass=float(model_output.unknown_mass[index].item()),
                content_entropy=float(entropy[index].item()),
                resolution_weighted_content_entropy=float(weighted_entropy[index].item()),
                resolution_entropy=float(resolution_entropy_value[index].item()),
                top1_class_mass=float(top1_class_mass[index].item()),
                proposition_truth_mass=float(proposition_truth_mass[index].item()),
                proposition_false_mass=float(proposition_false_mass[index].item()),
                proposition_unknown_mass=float(proposition_unknown_mass[index].item()),
                proposition_truth_ratio=float(proposition_truth_ratio[index].item()),
                ternary_entropy=float(ternary_entropy_value[index].item()),
                auxiliary_top1_content_probability=float(auxiliary_top1_content_probability[index].item()),
                completion_score_beta_0_1=float(score_beta_0_1[index].item()),
                completion_score_beta_0_25=float(score_beta_0_25[index].item()),
                completion_score_beta_0_5=float(score_beta_0_5[index].item()),
                completion_score_beta_0_75=float(score_beta_0_75[index].item()),
                candidate_class_indices=candidate_indices,
            )
        )
    return record_list


def build_top1_proposition_records(
    sample_analysis_records: Iterable[SampleAnalysisRecord],
) -> list[Top1PropositionRecord]:
    proposition_records: list[Top1PropositionRecord] = []
    for record in sample_analysis_records:
        if record.cohort_name == "ambiguous_id":
            is_top1_correct = record.predicted_class_index in record.candidate_class_indices
        elif record.cohort_name in {"easy_id", "hard_id"}:
            is_top1_correct = record.predicted_class_index == record.class_label
        else:
            is_top1_correct = False

        proposition_records.append(
            Top1PropositionRecord(
                model_family=record.model_family,
                run_id=record.run_id,
                protocol_id=record.protocol_id,
                sample_id=record.sample_id,
                split_name=record.split_name,
                cohort_name=record.cohort_name,
                proposition_target_type=_proposition_target_type(record.cohort_name),
                predicted_class_index=record.predicted_class_index,
                class_label=record.class_label,
                source_class_label=record.source_class_label,
                is_top1_correct=bool(is_top1_correct),
                proposition_truth_mass=record.proposition_truth_mass,
                proposition_false_mass=record.proposition_false_mass,
                proposition_unknown_mass=record.proposition_unknown_mass,
                proposition_truth_ratio=record.proposition_truth_ratio,
                resolution_entropy=record.resolution_entropy,
                ternary_entropy=record.ternary_entropy,
                auxiliary_top1_content_probability=record.auxiliary_top1_content_probability,
                candidate_class_indices=record.candidate_class_indices,
            )
        )
    return proposition_records


def run_inference_export(
    model: FRCNetModel,
    dataloader: DataLoader,
    runtime_spec,
    run_id: str,
    protocol_id: str,
    *,
    model_family: str = DEFAULT_MODEL_FAMILY,
) -> list[SampleAnalysisRecord]:
    sample_analysis_records: list[SampleAnalysisRecord] = []
    model.to(runtime_spec.device)
    model.eval()
    with torch.no_grad():
        for batch_input in dataloader:
            batch_on_device = move_batch_to_device(batch_input, runtime_spec)
            model_output = model(batch_on_device.image)
            sample_analysis_records.extend(
                build_sample_analysis_records(
                    model_output=model_output,
                    batch_input=batch_on_device,
                    run_id=run_id,
                    protocol_id=protocol_id,
                    model_family=model_family,
                )
            )
    return sample_analysis_records
