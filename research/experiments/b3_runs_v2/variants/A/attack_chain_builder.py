from neo4j_client import driver

ATTACK_FLOW = [
    "T1595",
    "T1110",
    "T1110.001",
    "T1078",
    "T1053.003"
]

def build_attack_chain():

    with driver.session() as session:

        for i in range(len(ATTACK_FLOW)-1):

            src = ATTACK_FLOW[i]
            dst = ATTACK_FLOW[i+1]

            session.run("""
                MATCH (a:Technique {attack_id: $src})
                MATCH (b:Technique {attack_id: $dst})
                MERGE (a)-[r:NEXT_TECHNIQUE]->(b)
                ON CREATE SET r.count = 1
                ON MATCH SET r.count = r.count + 1
            """,
            src=src,
            dst=dst)

if __name__ == "__main__":
    build_attack_chain()