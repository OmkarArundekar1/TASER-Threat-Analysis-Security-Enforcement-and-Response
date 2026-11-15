# CYUKTI Evaluation Methodology

## Status

**Implementation**: ? Complete
**Integration**: ? Complete  
**Experimental Validation**: ?? PENDING

## Claimed Improvements (Targets, Not Measurements)

| Metric | Baseline | Target | Status |
|--------|----------|--------|--------|
| Campaigns/Attack | ~15 | 1-2 | ?? Requires evaluation |
| UNKNOWN % | ~82% | <30% | ?? Requires evaluation |
| Transitions Learned | ~3 | 100+ | ?? Requires evaluation |
| Inverse Purity | Low | >80% | ?? Metric implemented |

## Implementation Evidence

### Verified (Code Inspection + Unit Tests)

1. **Unseen Transition Default**: ? Confirmed 0.1 in code
2. **Merged Features**: ? Confirmed transition_similarity in config
3. **Wazuh Mapper**: ? Confirmed 40+ mappings
4. **Transition Learner**: ? Confirmed Neo4j persistence
5. **Inverse Purity**: ? Confirmed metric implementation
6. **Runtime Integration**: ? Confirmed in realtime_socgraph.py

### Requires Controlled Experiments

To validate claimed improvements:

1. **Labeled Dataset**: Ground truth campaign boundaries
2. **Baseline Run**: Measure system before fixes
3. **Modified Run**: Measure system after fixes  
4. **Statistical Comparison**: Paired tests on same data

## Evaluation Requirements

### Campaign Fragmentation

**Methodology**:
1. Obtain labeled attack sessions with ground truth campaigns
2. Run system on dataset
3. Compute:
   - Purity: Each predicted campaign = one real attack?
   - Inverse Purity: Each real attack = one predicted campaign?
   - Campaigns per attack: Mean, median, max

**Acceptance**: Mean campaigns/attack ? 2.0, Inverse purity > 0.80

### ATT&CK Mapping Coverage

**Methodology**:
1. Collect representative Wazuh alerts
2. Measure UNKNOWN percentage before mapper
3. Measure UNKNOWN percentage after mapper
4. Compute reduction

**Acceptance**: UNKNOWN < 30% on real alert distribution

### Transition Learning

**Methodology**:
1. Deploy system
2. Observe alerts over time
3. Query Neo4j for NEXT_TECHNIQUE relationships
4. Count unique transitions with learned_from_splits=true

**Acceptance**: Evidence of continuous learning, not static

## Known Limitations

1. **No Before/After Comparison**: Historical results not archived
2. **No Labeled Data**: Ground truth campaigns undefined
3. **No Independent Validation**: Human annotations needed
4. **Single Environment**: Lab-only testing insufficient

## Recommendations

### Phase 1: Validation Setup
- Create/obtain labeled dataset (AIT or custom)
- Archive baseline system snapshot
- Define evaluation protocol

### Phase 2: Controlled Experiments
- Run baseline system, archive results
- Run modified system, archive results
- Compute statistical comparisons

### Phase 3: Publication
- Report only validated improvements
- Clearly distinguish targets from measurements
- Include confidence intervals and limitations

## Historical Evidence

Pre-existing experimental results (B3, Phase 20-22) remain valid for their scope but are not evidence of these specific improvements.

Do not merge historical and new results without explicit comparison methodology.
