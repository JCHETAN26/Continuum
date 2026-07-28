from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from continuum_linguistic_drift.schemas import (
    DocumentForAnalysis,
    EmergingTerm,
    EntityHit,
    LinguisticDriftReport,
    TopicShare,
)

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
ENTITY_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9_-]{2,})(?:\s+(?:[A-Z][A-Za-z0-9_-]{2,}))*\b")


class TopicBackend(Protocol):
    def fit_transform(self, documents: list[str]) -> tuple[list[int], object]: ...

    def get_topic_info(self) -> object: ...

    def get_topic(self, topic: int) -> list[tuple[str, float]] | bool: ...


@dataclass(frozen=True)
class LinguisticProfile:
    entity_counts: Counter[str]
    topic_distribution: dict[str, float]
    vocabulary_counts: Counter[str]
    documents: list[str]


class EntityExtractor:
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self._nlp = None
        self._loaded = False

    def extract(self, documents: list[str]) -> Counter[str]:
        if not self._loaded:
            self._nlp = self._load_spacy()
            self._loaded = True

        counts: Counter[str] = Counter()
        if self._nlp is not None:
            for doc in self._nlp.pipe(documents):
                for entity in doc.ents:
                    key = f"{entity.label_}:{entity.text.strip().lower()}"
                    counts[key] += 1
            return counts

        for text in documents:
            for match in ENTITY_RE.finditer(text):
                entity = match.group(0).strip()
                if entity.lower() not in {"the", "and"}:
                    counts[f"PROPER:{entity.lower()}"] += 1
        return counts

    def _load_spacy(self):
        try:
            import spacy

            return spacy.load(self.model_name)
        except Exception:
            return None


class TopicModeler:
    def __init__(self, min_topic_size: int = 5):
        self.min_topic_size = min_topic_size

    def distribution(self, documents: list[str]) -> dict[str, float]:
        if not documents:
            return {}

        backend = self._load_bertopic()
        if backend is not None and len(documents) >= self.min_topic_size:
            try:
                topics, _ = backend.fit_transform(documents)
                return self._bertopic_distribution(backend, topics)
            except Exception:
                pass

        return self._keyword_distribution(documents)

    def _load_bertopic(self) -> TopicBackend | None:
        try:
            from bertopic import BERTopic

            return BERTopic(min_topic_size=self.min_topic_size, calculate_probabilities=False)
        except Exception:
            return None

    def _bertopic_distribution(self, backend: TopicBackend, topics: list[int]) -> dict[str, float]:
        counts = Counter(topic for topic in topics if topic != -1)
        total = sum(counts.values())
        if total == 0:
            return {}

        distribution = {}
        for topic, count in counts.items():
            terms = backend.get_topic(topic) or []
            label = "_".join(term for term, _ in terms[:3]) or f"topic_{topic}"
            distribution[label] = count / total
        return distribution

    def _keyword_distribution(self, documents: list[str]) -> dict[str, float]:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=12, ngram_range=(1, 2))
        try:
            matrix = vectorizer.fit_transform(documents)
        except ValueError:
            return {}

        weights = np.asarray(matrix.sum(axis=0)).ravel()
        terms = vectorizer.get_feature_names_out()
        if float(weights.sum()) == 0:
            return {}

        top_indices = np.argsort(weights)[-6:]
        top_weights = weights[top_indices]
        total = float(top_weights.sum()) or 1.0
        return {terms[index]: float(weights[index] / total) for index in top_indices[::-1]}


class VocabularyShiftDetector:
    def counts(self, documents: list[str]) -> Counter[str]:
        counts: Counter[str] = Counter()
        for text in documents:
            counts.update(token.lower() for token in TOKEN_RE.findall(text))
        return counts

    def chi2_pvalue(self, baseline: Counter[str], window: Counter[str]) -> float:
        vocabulary = list((baseline | window).keys())
        if len(vocabulary) < 2:
            return 1.0

        baseline_total = sum(baseline.values())
        window_total = sum(window.values())
        if baseline_total == 0 or window_total == 0:
            return 1.0

        observed = np.array(
            [[baseline[term] for term in vocabulary], [window[term] for term in vocabulary]],
            dtype=float,
        )
        row_sums = observed.sum(axis=1)
        col_sums = observed.sum(axis=0)
        total = observed.sum()
        expected = np.outer(row_sums, col_sums) / total
        mask = expected > 0
        statistic = float(((observed[mask] - expected[mask]) ** 2 / expected[mask]).sum())
        degrees = max(1, len(vocabulary) - 1)

        from scipy.stats import chi2

        return float(chi2.sf(statistic, degrees))

    def emerging_terms(
        self, baseline: Counter[str], window: Counter[str], *, limit: int = 10
    ) -> list[EmergingTerm]:
        baseline_total = max(1, sum(baseline.values()))
        window_total = max(1, sum(window.values()))
        scored = []
        for term, window_count in window.items():
            baseline_count = baseline[term]
            window_rate = window_count / window_total
            baseline_rate = (baseline_count + 1) / (baseline_total + len(window))
            score = window_rate / baseline_rate
            if window_count >= 2 and score > 1.5:
                scored.append((score, term, baseline_count, window_count))

        return [
            EmergingTerm(
                term=term,
                baseline_count=baseline_count,
                window_count=window_count,
                score=round(float(score), 4),
            )
            for score, term, baseline_count, window_count in sorted(scored, reverse=True)[:limit]
        ]


class LinguisticDriftAnalyzer:
    def __init__(
        self,
        *,
        threshold: float = 0.65,
        entity_extractor: EntityExtractor | None = None,
        topic_modeler: TopicModeler | None = None,
        vocabulary_detector: VocabularyShiftDetector | None = None,
    ):
        self.threshold = threshold
        self.entity_extractor = entity_extractor or EntityExtractor()
        self.topic_modeler = topic_modeler or TopicModeler()
        self.vocabulary_detector = vocabulary_detector or VocabularyShiftDetector()

    def build_profile(self, documents: list[DocumentForAnalysis]) -> LinguisticProfile:
        texts = [doc.text for doc in documents if doc.text.strip()]
        return LinguisticProfile(
            entity_counts=self.entity_extractor.extract(texts),
            topic_distribution=self.topic_modeler.distribution(texts),
            vocabulary_counts=self.vocabulary_detector.counts(texts),
            documents=texts,
        )

    def analyze(
        self,
        baseline_documents: list[DocumentForAnalysis],
        window_documents: list[DocumentForAnalysis],
    ) -> LinguisticDriftReport:
        baseline = self.build_profile(baseline_documents)
        window = self.build_profile(window_documents)
        entity_kl = bounded_js_distance(baseline.entity_counts, window.entity_counts)
        topic_distance = topic_wasserstein(
            baseline.topic_distribution,
            window.topic_distribution,
            baseline.documents,
            window.documents,
        )
        pvalue = self.vocabulary_detector.chi2_pvalue(
            baseline.vocabulary_counts, window.vocabulary_counts
        )
        vocab_score = 1.0 - pvalue
        composite = clamp01(0.35 * entity_kl + 0.35 * topic_distance + 0.30 * vocab_score)

        return LinguisticDriftReport(
            document_count=len(window.documents),
            entity_kl_divergence=round(entity_kl, 6),
            topic_wasserstein=round(topic_distance, 6),
            vocab_chi2_pvalue=round(pvalue, 6),
            composite_score=round(composite, 6),
            threshold=self.threshold,
            breached=composite > self.threshold,
            new_entities=new_entities(baseline.entity_counts, window.entity_counts),
            emerging_topics=emerging_topics(baseline.topic_distribution, window.topic_distribution),
            emerging_terms=self.vocabulary_detector.emerging_terms(
                baseline.vocabulary_counts, window.vocabulary_counts
            ),
        )


def bounded_js_distance(left: Counter[str], right: Counter[str]) -> float:
    keys = sorted((left | right).keys())
    if not keys:
        return 0.0
    left_values = np.array([left[key] for key in keys], dtype=float)
    right_values = np.array([right[key] for key in keys], dtype=float)
    left_distribution = smooth_distribution(left_values)
    right_distribution = smooth_distribution(right_values)
    distance = float(jensenshannon(left_distribution, right_distribution, base=2.0))
    return clamp01(distance)


def topic_wasserstein(
    baseline_topics: dict[str, float],
    window_topics: dict[str, float],
    baseline_docs: list[str],
    window_docs: list[str],
) -> float:
    keys = sorted(set(baseline_topics) | set(window_topics))
    if keys:
        baseline = np.array([baseline_topics.get(key, 0.0) for key in keys], dtype=float)
        window = np.array([window_topics.get(key, 0.0) for key in keys], dtype=float)
        baseline_axis = np.arange(len(keys))
        window_axis = np.arange(len(keys))
        return clamp01(float(wasserstein_distance(baseline_axis, window_axis, baseline, window)))

    return document_centroid_distance(baseline_docs, window_docs)


def document_centroid_distance(baseline_docs: list[str], window_docs: list[str]) -> float:
    if not baseline_docs or not window_docs:
        return 0.0
    vectorizer = TfidfVectorizer(stop_words="english", max_features=128)
    matrix = vectorizer.fit_transform([*baseline_docs, *window_docs])
    baseline = np.asarray(matrix[: len(baseline_docs)].mean(axis=0))
    window = np.asarray(matrix[len(baseline_docs) :].mean(axis=0))
    similarity = float(cosine_similarity(baseline, window)[0][0])
    return clamp01(1.0 - similarity)


def smooth_distribution(values: np.ndarray) -> np.ndarray:
    smoothed = values + 1e-9
    total = float(smoothed.sum())
    if total == 0:
        return np.ones_like(smoothed) / len(smoothed)
    return smoothed / total


def new_entities(
    baseline: Counter[str], window: Counter[str], *, limit: int = 10
) -> list[EntityHit]:
    entities = []
    for key, count in window.items():
        if key not in baseline:
            label, _, text = key.partition(":")
            entities.append(EntityHit(text=text, label=label, count=count))
    return sorted(entities, key=lambda entity: entity.count, reverse=True)[:limit]


def emerging_topics(
    baseline: dict[str, float], window: dict[str, float], *, limit: int = 8
) -> list[TopicShare]:
    topics = []
    for label, share in window.items():
        if share - baseline.get(label, 0.0) > 0.05:
            topics.append(
                TopicShare(
                    label=label,
                    share=round(float(share), 6),
                    top_terms=[term for term in label.split("_") if term],
                )
            )
    return sorted(topics, key=lambda topic: topic.share, reverse=True)[:limit]


def clamp01(value: float) -> float:
    if math.isnan(value) or value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value
