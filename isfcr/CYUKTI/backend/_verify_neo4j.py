from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://localhost:7687', auth=('', ''))
with driver.session() as s:
    r1 = list(s.run('MATCH (t1:Technique)-[r:NEXT_TECHNIQUE]->(t2:Technique) RETURN t1.attack_id AS from_id, t2.attack_id AS to_id, r.count AS count'))
    print('=== NEXT_TECHNIQUE ===')
    for r in r1:
        print('  %s -> %s count=%s' % (r['from_id'], r['to_id'], r['count']))
    if not r1:
        print('  NONE FOUND')
    r2 = list(s.run('MATCH (c:Campaign)-[r:LIKELY_NEXT]->(t:Technique) RETURN c.campaign_id AS cid, t.attack_id AS tech, r.confidence AS conf, r.generated_at AS gen'))
    print('\n=== LIKELY_NEXT ===')
    for r in r2:
        print('  Campaign=%s -> %s conf=%s gen=%s' % (r['cid'], r['tech'], r['conf'], r['gen']))
    if not r2:
        print('  NONE FOUND')
    r3 = list(s.run('MATCH (t:Technique) RETURN keys(t) AS k LIMIT 3'))
    print('\n=== Technique Node Keys ===')
    for r in r3:
        print('  %s' % r['k'])
    print('\n=== Threat Intelligence Nodes ===')
    for label in ['ThreatActor', 'Malware', 'Tool', 'Mitigation', 'Tactic']:
        cnt = s.run('MATCH (n:%s) RETURN count(n) AS c' % label).single()['c']
        print('  %s: %d' % (label, cnt))
    r4 = list(s.run('MATCH (c:Campaign) RETURN c.campaign_id AS cid, c.last_technique AS lt, c.tps AS tps LIMIT 5'))
    print('\n=== Campaigns ===')
    for r in r4:
        print('  %s last_technique=%s tps=%s' % (r['cid'], r['lt'], r['tps']))
    r5 = s.run('MATCH (c:Campaign) RETURN max(c.tps) AS max_tps').single()
    print('\n=== Max Campaign TPS: %s ===' % r5['max_tps'])
    r6 = list(s.run('MATCH (e:AttackEvent) RETURN e.technique_id AS tid, e.campaign_id AS cid, e.occurrences AS occ, e.tps AS tps LIMIT 5'))
    print('\n=== AttackEvent Samples ===')
    for r in r6:
        print('  technique=%s campaign=%s occ=%s tps=%s' % (r['tid'], r['cid'], r['occ'], r['tps']))
driver.close()
print('\n=== VERIFICATION COMPLETE ===')
