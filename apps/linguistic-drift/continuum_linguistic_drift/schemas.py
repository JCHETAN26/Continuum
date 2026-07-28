from datetime import datetime

from pydantic import BaseModel, Field


class DocumentForAnalysis(BaseModel):
    id: str | None = None
    text: str
    source: str = "unknown"
    occurred_at: datetime | None = None


class EntityHit(BaseModel):
    text: str
    label: str
    count: int = Field(gt=0)


class TopicShare(BaseModel):
    label: str
    share: float = Field(ge=0.0, le=1.0)
    top_terms: list[str] = Field(default_factory=list)


class EmergingTerm(BaseModel):
    term: str
    baseline_count: int
    window_count: int
    score: float


class LinguisticDriftReport(BaseModel):
    document_count: int
    entity_kl_divergence: float = Field(ge=0.0)
    topic_wasserstein: float = Field(ge=0.0)
    vocab_chi2_pvalue: float = Field(ge=0.0, le=1.0)
    composite_score: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    breached: bool
    new_entities: list[EntityHit]
    emerging_topics: list[TopicShare]
    emerging_terms: list[EmergingTerm]
