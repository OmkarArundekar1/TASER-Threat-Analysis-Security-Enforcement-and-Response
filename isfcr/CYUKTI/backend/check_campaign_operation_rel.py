#!/usr/bin/env python3
from neo4j_client import driver

with driver.session() as session:
    # Check relationship type
    result = session.run('''
        MATCH (c:Campaign)-[r]->(o:Operation)
        RETURN type(r) as rel_type, count(*) as count
    ''')

    print("Campaign-Operation Relationships:")
    for record in result:
        print(f"  {record['rel_type']}: {record['count']}")
