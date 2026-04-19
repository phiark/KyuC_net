from __future__ import annotations

from frcnet.evaluation import build_top1_proposition_records
from frcnet.models import FRCNetModel
from frcnet.evaluation.inference import build_sample_analysis_records
from tests.conftest import build_synthetic_batch


def test_sample_analysis_records_expose_paper_fields():
    batch_input = build_synthetic_batch()
    model = FRCNetModel(num_classes=10)
    model_output = model(batch_input.image)

    records = build_sample_analysis_records(model_output, batch_input, run_id="RUN-1", protocol_id="plan_a_v1")

    assert len(records) == batch_input.batch_size
    first_record = records[0]
    assert first_record.run_id == "RUN-1"
    assert hasattr(first_record, "resolution_ratio")
    assert hasattr(first_record, "content_entropy")
    assert hasattr(first_record, "completion_score_beta_0_1")


def test_top1_proposition_records_filter_out_ood_and_unknown():
    batch_input = build_synthetic_batch()
    model = FRCNetModel(num_classes=10)
    model_output = model(batch_input.image)
    sample_records = build_sample_analysis_records(model_output, batch_input, run_id="RUN-1", protocol_id="plan_a_v1")

    proposition_records = build_top1_proposition_records(sample_records)

    assert len(proposition_records) == 3
    ambiguous_record = next(record for record in proposition_records if record.cohort_name == "ambiguous_id")
    assert ambiguous_record.proposition_target_type == "candidate_set"

