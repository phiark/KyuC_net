from __future__ import annotations

from typing import Iterable

import torch
from torch.utils.data import DataLoader

from frcnet.data import BatchInput
from frcnet.evaluation.records import SampleAnalysisRecord, Top1PropositionRecord
from frcnet.models import FRCNetModel, ModelOutput
from frcnet.utils import (
    completion_score,
    content_entropy,
    move_batch_to_device,
    resolution_weighted_content_entropy,
)


def build_sample_analysis_records(
    model_output: ModelOutput,
    batch_input: BatchInput,
    run_id: str,
    protocol_id: str,
) -> list[SampleAnalysisRecord]:
    predicted_class_index = torch.argmax(model_output.class_mass, dim=1)
    top1_content_probability = model_output.content_distribution.gather(
        1, predicted_class_index.unsqueeze(1)
    ).squeeze(1)
    record_list: list[SampleAnalysisRecord] = []
    entropy = content_entropy(model_output.content_distribution)
    weighted_entropy = resolution_weighted_content_entropy(model_output.resolution_ratio, entropy)
    score_beta_0_1 = completion_score(model_output.class_mass, model_output.unknown_mass, beta=0.1)
    score_beta_0_25 = completion_score(model_output.class_mass, model_output.unknown_mass, beta=0.25)
    score_beta_0_5 = completion_score(model_output.class_mass, model_output.unknown_mass, beta=0.5)
    score_beta_0_75 = completion_score(model_output.class_mass, model_output.unknown_mass, beta=0.75)
    top1_class_mass = model_output.class_mass.gather(1, predicted_class_index.unsqueeze(1)).squeeze(1)

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
                top1_class_mass=float(top1_class_mass[index].item()),
                top1_content_probability=float(top1_content_probability[index].item()),
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
        if record.cohort_name in {"ood", "unknown_supervision"}:
            continue

        if record.cohort_name == "ambiguous_id":
            proposition_target_type = "candidate_set"
            is_top1_correct = record.predicted_class_index in record.candidate_class_indices
        else:
            proposition_target_type = "single_label"
            is_top1_correct = record.predicted_class_index == record.class_label

        proposition_records.append(
            Top1PropositionRecord(
                run_id=record.run_id,
                protocol_id=record.protocol_id,
                sample_id=record.sample_id,
                split_name=record.split_name,
                cohort_name=record.cohort_name,
                proposition_target_type=proposition_target_type,
                predicted_class_index=record.predicted_class_index,
                class_label=record.class_label,
                source_class_label=record.source_class_label,
                is_top1_correct=bool(is_top1_correct),
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
                )
            )
    return sample_analysis_records
