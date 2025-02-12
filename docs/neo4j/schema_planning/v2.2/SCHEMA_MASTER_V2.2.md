# Dell-AITC Schema Documentation v2.2

## Overview
This document describes the Neo4j database schema for the Dell-AITC system version 2.2, focusing on AI technology classification and federal use case tracking.

## Version Information
```yaml
Version: 2.2
Status: Active
Last Updated: 2024-02
Changes:
  - Restored AIClassification node with enhanced metadata
  - Added NoMatchAnalysis node for unmatched cases
  - Improved LLM analysis integration
  - Maintained all v2.1.2 organizational structures
```

## Node Types

### AICategory
Represents an AI technology category.

Properties:
- `id` (uuid, required): Unique identifier
- `name` (string, required, unique): Category name
- `category_definition` (text, required): Detailed definition
- `status` (enum): Active, Deprecated, Proposed
- `maturity_level` (enum): Emerging, Established, Mature
- `zone_id` (uuid): Reference to technology zone
- `created_at` (datetime): Creation timestamp
- `last_updated` (datetime): Last update timestamp
- `version` (string): Version identifier

### Zone
Represents a technology zone grouping AI categories.

Properties:
- `id` (uuid, required): Unique identifier
- `name` (string, required, unique): Zone name
- `description` (string): Zone description
- `created_at` (datetime): Creation timestamp
- `last_updated` (datetime): Last update timestamp

### Keyword
Represents technical keywords for AI categories.

Properties:
- `id` (uuid, required): Unique identifier
- `name` (string, required, unique): Keyword text
- `type` (string): Keyword type classification
- `relevance_score` (float): Relevance score
- `created_at` (datetime): Creation timestamp
- `last_updated` (datetime): Last update timestamp

### UseCase
Represents a federal use case of AI technology.

Properties:
- `id` (uuid, required): Unique identifier
- `name` (string, required): Use case name
- `topic_area` (enum): Business domain
- `stage` (enum): Development stage
- `impact_type` (enum): Impact classification
- `purpose_benefits` (text): Purpose and benefits
- `outputs` (text): Expected outputs
- `dev_method` (enum): Development methodology
- `contains_pii` (boolean): PII indicator
- `has_ato` (boolean): ATO status
- `system_name` (string): System identifier
- `date_initiated` (datetime): Project start
- `date_acquisition` (datetime): Acquisition date
- `date_implemented` (datetime): Implementation date
- `date_retired` (datetime): Retirement date
- `created_at` (datetime): Creation timestamp
- `last_updated` (datetime): Last update timestamp

### Agency
Represents a federal agency.

Properties:
- `id` (uuid, required): Unique identifier
- `name` (string, required, unique): Agency name
- `abbreviation` (string, required, unique): Agency abbreviation
- `created_at` (datetime): Creation timestamp
- `last_updated` (datetime): Last update timestamp

### Bureau
Represents an agency bureau or sub-organization.

Properties:
- `id` (uuid, required): Unique identifier
- `name` (string, required): Bureau name
- `agency_id` (uuid, required): Parent agency reference
- `created_at` (datetime): Creation timestamp
- `last_updated` (datetime): Last update timestamp

### AIClassification
Represents the classification of use cases into AI technology categories.

Properties:
- `id` (uuid, required): Unique identifier
- `match_type` (enum, required): PRIMARY, SUPPORTING, RELATED
- `match_rank` (integer): Order of matches
- `confidence` (float, required): Overall confidence score (0.0-1.0)
- `analysis_method` (enum, required): KEYWORD, SEMANTIC, LLM, ENSEMBLE
- `analysis_version` (string): Algorithm version
- `keyword_score` (float): Keyword analysis score
- `semantic_score` (float): Semantic analysis score
- `llm_score` (float): LLM analysis score
- `field_match_scores` (json): Field-level match details
- `term_match_details` (json): Term match specifics
- `matched_keywords` (string[]): Matched keywords
- `llm_verification` (boolean): LLM verification status
- `llm_confidence` (float): LLM confidence score
- `llm_reasoning` (string): LLM explanation
- `llm_suggestions` (json): LLM improvement suggestions
- `improvement_notes` (string[]): Notes for improvement
- `false_positive` (boolean): False positive flag
- `manual_override` (boolean): Manual override flag
- `review_status` (enum): PENDING, REVIEWED, VERIFIED
- `classified_at` (datetime): Classification timestamp
- `classified_by` (string): Classifier identifier
- `last_updated` (datetime): Last update timestamp

### NoMatchAnalysis
Records analysis of use cases that don't match existing categories.

Properties:
- `id` (uuid, required): Unique identifier
- `reason` (string, required): Rejection reason
- `confidence` (float): No-match confidence
- `llm_analysis` (json): LLM analysis details
- `suggested_keywords` (string[]): Suggested keywords
- `improvement_suggestions` (json): Improvement ideas
- `created_at` (datetime): Creation timestamp
- `analyzed_by` (string): Analyzer identifier
- `status` (enum): NEW, REVIEWED, ACTIONED
- `review_notes` (string): Review notes

## Relationships

### CLASSIFIES
Connects AI categories to their classifications.

Properties:
- `created_at` (datetime): Creation timestamp

### CLASSIFIED_AS
Connects use cases to their classifications.

Properties:
- `created_at` (datetime): Creation timestamp

### HAS_ANALYSIS
Connects use cases to their no-match analysis.

Properties:
- `created_at` (datetime): Creation timestamp

### SUGGESTS_CATEGORY
Connects no-match analysis to suggested AI categories.

Properties:
- `created_at` (datetime): Creation timestamp
- `confidence` (float): Suggestion confidence

### HAS_BUREAU
Connects agencies to their bureaus.

Properties:
- `created_at` (datetime): Creation timestamp

### IMPLEMENTED_BY
Connects use cases to implementing agencies.

Properties:
- `created_at` (datetime): Creation timestamp

### MANAGED_BY
Connects use cases to managing bureaus.

Properties:
- `created_at` (datetime): Creation timestamp

### BELONGS_TO
Connects AI categories to technology zones.

Properties:
- `created_at` (datetime): Creation timestamp
- `weight` (float): Relationship strength

### HAS_KEYWORD
Connects AI categories to keywords.

Properties:
- `relevance` (float): Keyword relevance score
- `created_at` (datetime): Creation timestamp

### DEPENDS_ON
Represents dependencies between AI categories.

Properties:
- `strength` (float): Dependency strength
- `created_at` (datetime): Creation timestamp

## Constraints
1. Unique use case IDs
2. Unique category names
3. Unique agency names and abbreviations
4. Unique zone names
5. Unique keyword names
6. Valid bureau agency references
7. Single PRIMARY classification per use case
8. Valid match_type values
9. Valid review_status values

## Indexes
1. B-tree index on UseCase.id
2. B-tree index on AICategory.name
3. B-tree index on Agency.name
4. B-tree index on Agency.abbreviation
5. B-tree index on Zone.name
6. B-tree index on Keyword.name
7. B-tree index on AIClassification.match_type
8. B-tree index on NoMatchAnalysis.status
9. Full-text index on UseCase(name, description, purpose_benefits)
10. Full-text index on AICategory.definition
11. Full-text index on Keyword.name

## Query Optimization
1. Use case text search via fulltext indexes
2. Category matching via keyword and semantic indexes
3. Agency hierarchy traversal optimization
4. Impact analysis queries

## Implementation Notes
1. Use UUIDs for all IDs
2. Maintain audit timestamps
3. Validate all enums
4. Check referential integrity 