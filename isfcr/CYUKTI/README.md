# CYUKTI - Campaign Intelligence Framework

**Status**: Engineering Closure Complete (Nov 2024)

## Overview

CYUKTI is a graph-based cybersecurity campaign intelligence framework that transforms isolated security alerts into actionable attack campaign narratives.

### Core Capabilities

- Campaign correlation and lifecycle management
- Multi-tier ATT&CK technique resolution  
- Cross-campaign transition learning
- Temporal attack graph construction
- Threat intelligence integration
- MISP event publishing

## Recent Improvements

### Campaign Scoring Fixes
- Unseen transitions: 0.1 default (prevents fragmentation)
- Merged features: transition = prediction + chain
- Enhanced metrics: inverse purity, campaigns-per-attack

### ATT&CK Mapping
- Curated Wazuh mapper integrated (40+ rules)
- Reduces UNKNOWN labeling rate
- Extensible for environment-specific mappings

### Cross-Campaign Learning
- Transition learner records all attacker behavior
- Learns across campaign splits
- Neo4j persistence with provenance

## Documentation

- **docs/CRITICAL_FIXES.md** - Implementation details
- **docs/DEPLOYMENT.md** - Setup guide  
- **docs/EVALUATION.md** - Research methodology

## Quick Start

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
python realtime_socgraph.py
```

## Status

? Implementation complete
? Runtime integration complete
?? Experimental validation pending

See docs/ for details.
