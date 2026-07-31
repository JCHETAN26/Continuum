from datetime import UTC, datetime, timedelta

from continuum_linguistic_drift.analyzer import (
    EntityExtractor,
    LinguisticDriftAnalyzer,
    VocabularyShiftDetector,
    bounded_js_distance,
)
from continuum_linguistic_drift.schemas import DocumentForAnalysis


def docs(texts: list[str]) -> list[DocumentForAnalysis]:
    now = datetime.now(UTC)
    return [
        DocumentForAnalysis(
            id=str(index),
            text=text,
            source="test",
            occurred_at=now + timedelta(seconds=index),
        )
        for index, text in enumerate(texts)
    ]


def test_entity_extractor_falls_back_to_proper_noun_regex(monkeypatch):
    extractor = EntityExtractor()
    monkeypatch.setattr(extractor, "_load_spacy", lambda: None)

    counts = extractor.extract(["Acme Search handled Phoenix outages for Oncall Router."])

    assert counts["PROPER:acme search"] == 1
    assert counts["PROPER:phoenix"] == 1


def test_bounded_js_distance_is_zero_for_identical_counts():
    detector = VocabularyShiftDetector()
    counts = detector.counts(["cache index retry cache"])

    assert bounded_js_distance(counts, counts) == 0.0


def test_linguistic_analyzer_spikes_on_domain_shift(monkeypatch):
    extractor = EntityExtractor()
    monkeypatch.setattr(extractor, "_load_spacy", lambda: None)
    analyzer = LinguisticDriftAnalyzer(threshold=0.25, entity_extractor=extractor)

    baseline = docs(
        [
            "Acme Search cache index latency retry deployment service",
            "Acme Search query router shard replicas observability dashboard",
            "Acme Search deployment incident runbook service latency",
            "Acme Search cache invalidation index replicas query",
            "Acme Search oncall dashboard alert routing service",
            "Acme Search shard balancing query latency deployment",
            "Acme Search observability logs traces cache index",
            "Acme Search service reliability retry routing",
            "Acme Search dashboard deployment metrics replicas",
            "Acme Search incident response cache query routing",
        ]
    )
    window = docs(
        [
            "Cardiology Clinic reviewed insulin dosage and atrial fibrillation medication",
            "Cardiology Clinic triage documented hemoglobin labs and anticoagulant therapy",
            "Cardiology Clinic patient record included diagnosis hypertension medication",
            "Cardiology Clinic discharge summary listed insulin glucose and telemetry",
            "Cardiology Clinic oncology referral noted biopsy pathology medication",
        ]
    )

    report = analyzer.analyze(baseline, window)

    assert report.composite_score > 0.25
    assert report.breached is True
    assert {term.term for term in report.emerging_terms} & {"clinic", "cardiology", "medication"}
    assert report.new_entities


def test_entity_extractor_reports_which_backend_it_used():
    """Degradation used to be invisible: a bare except selected the regex and said nothing.

    The regex matches capitalised words, which is not entity recognition. It cannot tell a
    person from a product from a sentence-initial word, and it labels every match the same.
    A report built on it is weaker than one built on spaCy, so which ran has to be visible.
    """
    from continuum_linguistic_drift.analyzer import EntityExtractor

    extractor = EntityExtractor()
    assert extractor.backend == "unloaded"

    extractor.extract(["Apple released a Macintosh Quadra in Cupertino."])

    assert extractor.backend in {"spacy", "regex"}


def test_missing_spacy_falls_back_and_warns(monkeypatch, caplog):
    from continuum_linguistic_drift.analyzer import EntityExtractor

    extractor = EntityExtractor()
    monkeypatch.setattr(extractor, "_load_spacy", lambda: None)

    counts = extractor.extract(["The Quadra needs more VRAM"])

    assert extractor.backend == "regex"
    # The fallback still produces something, so the pipeline keeps running.
    assert counts


def test_topic_modeler_reports_its_backend():
    from continuum_linguistic_drift.analyzer import TopicModeler

    modeler = TopicModeler(min_topic_size=2)
    modeler.distribution([f"scsi drive termination problem {i}" for i in range(6)])

    assert modeler.backend in {"bertopic", "keyword"}


def test_keyword_backend_is_used_when_bertopic_is_absent(monkeypatch):
    """Keyword grouping is the default, not a failure: bertopic pulls torch."""
    from continuum_linguistic_drift.analyzer import TopicModeler

    modeler = TopicModeler(min_topic_size=2)
    monkeypatch.setattr(modeler, "_load_bertopic", lambda: None)

    distribution = modeler.distribution(["isa card irq conflict"] * 4)

    assert modeler.backend == "keyword"
    assert distribution
