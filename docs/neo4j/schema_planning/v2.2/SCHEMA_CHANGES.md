# Schema Changes v2.2 - Enhanced Technology Classification

## Overview
This schema update enhances the relationship between use cases and AI technologies to better support:
1. Primary/Supporting/Related technology classification
2. LLM analysis metadata
3. Continuous improvement tracking

## Changes

### 1. USES_TECHNOLOGY Relationship Updates
- **Match Type Classification**
  - Add `match_type` property (enum: PRIMARY, SUPPORTING, RELATED)
  - Add `match_rank` property (integer) for ordering multiple matches
  - Update `confidence` property validation (0.0-1.0)

- **Analysis Method Properties**
  - Add `analysis_method` property (enum: KEYWORD, SEMANTIC, LLM, ENSEMBLE)
  - Add `analysis_version` property (string) for tracking algorithm versions

- **Score Components**
  - Existing: `keyword_score`, `semantic_score`, `llm_score`
  - Add `field_match_scores` property (JSON) for detailed field-level matching
  - Add `term_match_details` property (JSON) for keyword/term match specifics

- **LLM Analysis Metadata**
  - Add `llm_verification` property (boolean)
  - Add `llm_confidence` property (float)
  - Add `llm_reasoning` property (string)
  - Add `llm_suggestions` property (JSON array)

- **Improvement Tracking**
  - Add `improvement_notes` property (string array)
  - Add `false_positive` property (boolean)
  - Add `manual_override` property (boolean)
  - Add `review_status` property (enum: PENDING, REVIEWED, VERIFIED)

### 2. New NO_MATCH Relationship
```cypher
(:UseCase)-[:NO_MATCH]->(:AICategory)
```
Properties:
- `reason` (string) - Why the match was rejected
- `confidence` (float) - Confidence in the no-match decision
- `llm_analysis` (JSON) - Detailed LLM analysis of why no match
- `suggested_keywords` (string array) - Keywords that might help match
- `improvement_suggestions` (JSON) - Suggestions for improving matching

## Migration Steps

### Phase 1: Schema Preparation
1. Create new property constraints and indexes
```cypher
CREATE CONSTRAINT unique_use_case_primary_match IF NOT EXISTS
FOR ()-[r:USES_TECHNOLOGY]->()
WHERE r.match_type = 'PRIMARY'
ASSERT r IS UNIQUE;
```

### Phase 2: Data Migration
1. Backup existing relationships
```cypher
MATCH (u:UseCase)-[r:USES_TECHNOLOGY]->(c:AICategory)
WITH u, c, properties(r) as props
MERGE (u)-[new:USES_TECHNOLOGY_V2]->(c)
SET new = props;
```

2. Update existing relationships with new properties
```cypher
MATCH ()-[r:USES_TECHNOLOGY]->()
SET r.match_type = 
  CASE 
    WHEN r.confidence >= 0.45 THEN 'PRIMARY'
    WHEN r.confidence >= 0.35 THEN 'SUPPORTING'
    ELSE 'RELATED'
  END,
  r.analysis_version = 'v2.2',
  r.review_status = 'PENDING';
```

### Phase 3: Validation
1. Verify data integrity
2. Check constraint violations
3. Validate relationship properties
4. Test queries with new schema

## Rollback Plan
1. Keep backup relationships (USES_TECHNOLOGY_V2) for 1 week
2. Maintain rollback scripts
3. Document validation queries

## Impact Analysis

### Performance Impact
- New indexes on `match_type` and `review_status`
- Additional properties increase relationship size
- New JSON properties require more storage

### Query Impact
- Update existing queries to handle new properties
- Add new queries for match type filtering
- Enhance aggregation queries

### Application Impact
- Update classifier service to set new properties
- Modify API responses to include new fields
- Update UI to display match types

## Testing Requirements
1. Verify unique primary match constraint
2. Test multiple supporting/related matches
3. Validate LLM analysis storage
4. Check migration script performance
5. Verify rollback procedures

## Documentation Updates Required
1. Update schema visualization
2. Update API documentation
3. Update query examples
4. Update application configuration

## Timeline
1. Development & Testing: 2-3 days
2. Migration: 1-2 hours (off-peak)
3. Validation: 1 day
4. Rollback window: 1 week 