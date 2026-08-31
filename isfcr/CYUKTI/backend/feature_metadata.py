from dataclasses import dataclass

@dataclass
class FeatureMetadata:
    confidence: float
    implemented: bool
    source: str
    
@dataclass
class FeatureValue:
    value: float | None
    confidence: float
    implemented: bool
    source: str

def implemented(value, source, confidence=1.0):
    return FeatureValue(
        value=value,
        confidence=confidence,
        implemented=True,
        source=source
    )


def unavailable(source):
    return FeatureValue(
        value=None,
        confidence=0.0,
        implemented=False,
        source=source
    )