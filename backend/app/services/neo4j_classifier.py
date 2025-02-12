from typing import List, Optional

class Neo4jClassifier:
    def get_unclassified_use_cases(self, limit: Optional[int] = None) -> List[dict]:
        """Fetch use cases without technology classifications"""
        query = """
        MATCH (u:UseCase)
        OPTIONAL MATCH (u)-[:IMPLEMENTED_BY]->(a:Agency)
        WHERE NOT EXISTS((u)-[:USES_TECHNOLOGY]->(:AICategory))
        RETURN {
            id: u.id,
            name: u.name,
            description: u.description,
            purpose_benefits: u.purpose_benefits,
            outputs: u.outputs,
            agency: CASE 
                WHEN a IS NOT NULL 
                THEN {
                    name: a.name,
                    abbreviation: COALESCE(a.abbreviation, 'Unknown')
                }
                ELSE {
                    name: 'Unknown Agency',
                    abbreviation: 'Unknown'
                }
            END
        }
        """ + (f" LIMIT {limit}" if limit else "")

        with self.driver.session() as session:
            result = session.run(query)
            return [record[0] for record in result] 