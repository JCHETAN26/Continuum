import os
from datetime import UTC, datetime

import pytest
from continuum_linguistic_drift.analyzer import EntityExtractor, LinguisticDriftAnalyzer
from continuum_linguistic_drift.schemas import DocumentForAnalysis

pytestmark = pytest.mark.integration


def build_docs(texts: list[str], source: str) -> list[DocumentForAnalysis]:
    now = datetime.now(UTC)
    return [
        DocumentForAnalysis(id=f"{source}-{index}", text=text, source=source, occurred_at=now)
        for index, text in enumerate(texts)
    ]


@pytest.mark.skipif(
    os.getenv("RUN_LINGUISTIC_INTEGRATION") != "1",
    reason="Set RUN_LINGUISTIC_INTEGRATION=1 to run linguistic drift integration.",
)
def test_linguistic_drift_detects_healthcare_shift(monkeypatch):
    extractor = EntityExtractor()
    monkeypatch.setattr(extractor, "_load_spacy", lambda: None)
    analyzer = LinguisticDriftAnalyzer(threshold=0.6, entity_extractor=extractor)
    software = build_docs(
        [
            "Acme Search cache index deployment query router incident metrics",
            "Acme Search Kubernetes service latency dashboard runbook",
            "Acme Search PostgreSQL index Redis cache observability",
            "Acme Search TypeScript API worker deployment pipeline",
            "Acme Search query shard replica logs tracing service",
        ]
        * 20,
        "github_issues",
    )
    healthcare = build_docs(
        [
            "Cardiology Clinic insulin dosage hypertension patient medication",
            "Cardiology Clinic echocardiogram atrial fibrillation diagnosis",
            "Cardiology Clinic oncology biopsy pathology treatment",
            "Cardiology Clinic hemoglobin anticoagulant therapy vitals",
            "Cardiology Clinic respiratory distress losartan prescription",
        ]
        * 10,
        "medical_records",
    )

    report = analyzer.analyze(software, healthcare)

    assert report.composite_score > 0.6
    assert report.breached is True
    assert report.new_entities
    assert {term.term for term in report.emerging_terms} & {
        "cardiology",
        "clinic",
        "patient",
        "medication",
    }
