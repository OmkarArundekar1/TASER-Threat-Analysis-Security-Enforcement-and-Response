"""
evaluation
============
CYUKTI's independent ground-truth and evaluation framework.

Nothing in this package is allowed to derive a ground-truth label from
CYUKTI's own output (mitre_resolver, campaign_manager,
threat_attribution_engine, threat_qualification, prediction_engine, or
any RAG retriever). See ground_truth/schema.py's module docstring for
the enforced boundary and evaluation/PRINCIPLES.md for the full
anti-circularity rationale.
"""
