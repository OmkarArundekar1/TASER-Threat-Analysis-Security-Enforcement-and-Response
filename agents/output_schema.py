"""
agents/output_schema.py
========================
Pydantic v2 schemas for structured JSON output from the SOC pipeline.

All fields are JSON-serialisable (no torch tensors, no numpy arrays).
These schemas act as the contract between the detection pipeline and
downstream LangChain agents / RAG system / dashboards.
"""

from __future__ import annotations

from typing import Any

try:
    from pydantic import BaseModel, Field
    _HAS_PYDANTIC = True
except ImportError:
    _HAS_PYDANTIC = False
    # Minimal fallback if pydantic is not installed
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def model_dump(self):
            return {k: getattr(self, k) for k in self.__class__.__annotations__}
    def Field(default=None, **kwargs):
        return default


class AnomalyResult(BaseModel):
    """Output from the SOC anomaly scorer."""
    window_index: int = Field(default=0, description="Index of the sliding window")
    severity_score: float = Field(description="Raw MAD-normalised severity score")
    severity_label: str = Field(description="NORMAL | LOW | MEDIUM | HIGH")
    temporal_label: str = Field(description="Temporally smoothed label")
    smoothed_score: float = Field(description="EMA-smoothed severity score")
    is_anomaly: int = Field(description="Binary: 1=anomaly, 0=normal")
    confidence: float = Field(description="Sigmoid confidence in [0, 1]")
    top_features: list[dict[str, Any]] = Field(default_factory=list)


class ClassificationResult(BaseModel):
    """Output from the LightGBM attack classifier."""
    label: str = Field(description="Predicted attack label")
    confidence: float = Field(description="Classifier confidence")
    probabilities: dict[str, float] = Field(default_factory=dict)
    mitre_tactic: str = Field(default="Unknown")
    kill_chain_phase: str = Field(default="Unknown")
    risk_level: str = Field(default="LOW")


class KillChainResult(BaseModel):
    """Kill-chain campaign inference result."""
    detected_stages: list[str] = Field(default_factory=list)
    current_phase: str = Field(default="None")
    progression_score: float = Field(default=0.0)
    is_active_campaign: bool = Field(default=False)
    is_multi_stage: bool = Field(default=False)
    recommended_priority: str = Field(default="P4_LOW")
    event_count: int = Field(default=0)


class ThreatIntelReport(BaseModel):
    """
    Full structured threat intelligence report — the main pipeline output.

    JSON-ready, LangChain-compatible, SIEM-ingestible.
    """
    # Identity
    report_id: str = Field(description="Unique report UUID")
    timestamp: str = Field(description="ISO 8601 analysis timestamp")
    sensor: str = Field(default="cicids")

    # Anomaly detection
    window_index: int = Field(default=0)
    severity_score: float = Field(default=0.0)
    severity_label: str = Field(default="NORMAL")
    temporal_label: str = Field(default="NORMAL")
    smoothed_score: float = Field(default=0.0)
    is_anomaly: int = Field(default=0)
    confidence: float = Field(default=0.0)

    # Explainability
    top_features: list[dict[str, Any]] = Field(default_factory=list)

    # Attack classification
    attack_label: str = Field(default="BENIGN")
    attack_confidence: float = Field(default=0.0)
    mitre_tactic: str = Field(default="None")
    kill_chain_phase: str = Field(default="None")
    risk_level: str = Field(default="NONE")

    # Kill-chain analysis
    is_active_campaign: bool = Field(default=False)
    progression_score: float = Field(default=0.0)
    recommended_priority: str = Field(default="P4_LOW")

    # Response
    playbook_id: str = Field(default="")
    recommended_actions: list[dict[str, Any]] = Field(default_factory=list)

    # SSFT preparation
    ssft_label: int = Field(default=0, description="0=benign, 1=attack (for fine-tuning)")

    def to_dict(self) -> dict[str, Any]:
        if _HAS_PYDANTIC:
            return self.model_dump()
        return {k: getattr(self, k) for k in self.__annotations__}
