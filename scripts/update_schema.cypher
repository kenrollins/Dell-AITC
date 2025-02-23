// Update Keyword nodes to include relevance_score
MATCH (k:Keyword)
WHERE k.relevance_score IS NULL
SET k.relevance_score = 0.5;

// Update HAS_KEYWORD relationships to include relevance
MATCH (c:AICategory)-[r:HAS_KEYWORD]->(k:Keyword)
WHERE r.relevance IS NULL
SET r.relevance = 0.5;

// Create index for relevance_score
CREATE INDEX keyword_relevance IF NOT EXISTS
FOR (k:Keyword) ON (k.relevance_score);

// Create index for relationship relevance
CREATE INDEX has_keyword_relevance IF NOT EXISTS
FOR ()-[r:HAS_KEYWORD]-() ON (r.relevance); 