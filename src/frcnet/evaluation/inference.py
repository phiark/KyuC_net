from __future__ import annotations

from typing import Iterable

import torch
from torch.utils.data import DataLoader

from frcnet.data import BatchInput
from frcnet.evaluation.beta_policy import completion_from_view
from frcnet.evaluation.proposition_views import proposition_view_from_mask, top1_view
from frcnet.evaluation.records import (
    DEFAULT_MODEL_FAMILY,
    PropositionViewRecord,
    SampleAnalysisRecord,
    Top1PropositionRecord,
)
from frcnet.evaluation.state_metrics import (
    state_content_entropy,
    state_entropy,
    state_weighted_content_entropy,
)
from frcnet.models import FRCNetModel, ModelOutput
from frcnet.utils import (
    move_batch_to_device,
    resolution_entropy,
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
    entropy = state_content_entropy(model_output.content_distribution)
    weighted_entropy = state_weighted_content_entropy(model_output.resolution_ratio, entropy)
    state_entropy_value = state_entropy(model_output.resolution_ratio, entropy)
    resolution_entropy_value = resolution_entropy(model_output.resolution_ratio)
    top1_class_mass = model_output.class_mass.gather(1, predicted_class_index.unsqueeze(1)).squeeze(1)
    top1_view_value = top1_view(model_output.class_mass, model_output.unknown_mass)
    score_beta_0_1 = completion_from_view(top1_view_value.truth_mass, top1_view_value.unknown_mass, beta=0.1)
    score_beta_0_25 = completion_from_view(top1_view_value.truth_mass, top1_view_value.unknown_mass, beta=0.25)
    score_beta_0_5 = completion_from_view(top1_view_value.truth_mass, top1_view_value.unknown_mass, beta=0.5)
    score_beta_0_75 = completion_from_view(top1_view_value.truth_mass, top1_view_value.unknown_mass, beta=0.75)
    proposition_target_mask = _build_proposition_target_mask(
        batch_input,
        model_output.num_classes,
        model_output.class_mass.device,
    )
    target_view = proposition_view_from_mask(
        view_name="target_or_candidate_view",
        label_aware=True,
        class_mass=model_output.class_mass,
        unknown_mass=model_output.unknown_mass,
        truth_mask=proposition_target_mask,
    )
    ternary_entropy_value = ternary_entropy_from_masses(
        target_view.truth_mass,
        target_view.false_mass,
        target_view.unknown_mass,
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
        source_dataset_split = "" if batch_input.source_dataset_split is None else batch_input.source_dataset_split[index]
        source_role = "" if batch_input.source_role is None else batch_input.source_role[index]
        source_partition_name = (
            "" if batch_input.source_partition_name is None else batch_input.source_partition_name[index]
        )
        source_sample_indices = (
            ()
            if batch_input.source_sample_indices is None
            else tuple(int(value) for value in batch_input.source_sample_indices[index])
        )
        augmentation_recipe = "" if batch_input.augmentation_recipe is None else batch_input.augmentation_recipe[index]

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
                state_content_entropy=float(entropy[index].item()),
                state_weighted_content_entropy=float(weighted_entropy[index].item()),
                state_entropy=float(state_entropy_value[index].item()),
                resolution_entropy=float(resolution_entropy_value[index].item()),
                top1_class_mass=float(top1_class_mass[index].item()),
                top1_view_truth_mass=float(top1_view_value.truth_mass[index].item()),
                top1_view_false_mass=float(top1_view_value.false_mass[index].item()),
                top1_view_unknown_mass=float(top1_view_value.unknown_mass[index].item()),
                top1_view_tau=float(top1_view_value.tau[index].item()),
                proposition_truth_mass=float(target_view.truth_mass[index].item()),
                proposition_false_mass=float(target_view.false_mass[index].item()),
                proposition_unknown_mass=float(target_view.unknown_mass[index].item()),
                proposition_truth_ratio=float(target_view.tau[index].item()),
                ternary_entropy=float(ternary_entropy_value[index].item()),
                auxiliary_top1_content_probability=float(auxiliary_top1_content_probability[index].item()),
                top1_completion_beta_0_1=float(score_beta_0_1[index].item()),
                top1_completion_beta_0_25=float(score_beta_0_25[index].item()),
                top1_completion_beta_0_5=float(score_beta_0_5[index].item()),
                top1_completion_beta_0_75=float(score_beta_0_75[index].item()),
                candidate_class_indices=candidate_indices,
                source_dataset_split=source_dataset_split,
                source_role=source_role,
                source_partition_name=source_partition_name,
                source_sample_indices=source_sample_indices,
                augmentation_recipe=augmentation_recipe,
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


def build_proposition_view_records(
    sample_analysis_records: Iterable[SampleAnalysisRecord],
) -> list[PropositionViewRecord]:
    proposition_view_records: list[PropositionViewRecord] = []
    for record in sample_analysis_records:
        proposition_view_records.append(
            PropositionViewRecord(
                model_family=record.model_family,
                run_id=record.run_id,
                protocol_id=record.protocol_id,
                sample_id=record.sample_id,
                split_name=record.split_name,
                cohort_name=record.cohort_name,
                view_name="top1_view",
                label_aware=False,
                proposition_truth_mass=record.top1_view_truth_mass,
                proposition_false_mass=record.top1_view_false_mass,
                proposition_unknown_mass=record.top1_view_unknown_mass,
                proposition_truth_ratio=record.top1_view_tau,
            )
        )
        proposition_view_records.append(
            PropositionViewRecord(
                model_family=record.model_family,
                run_id=record.run_id,
                protocol_id=record.protocol_id,
                sample_id=record.sample_id,
                split_name=record.split_name,
                cohort_name=record.cohort_name,
                view_name=f"{_proposition_target_type(record.cohort_name)}_view",
                label_aware=True,
                proposition_truth_mass=record.proposition_truth_mass,
                proposition_false_mass=record.proposition_false_mass,
                proposition_unknown_mass=record.proposition_unknown_mass,
                proposition_truth_ratio=record.proposition_truth_ratio,
            )
        )
    return proposition_view_records


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
