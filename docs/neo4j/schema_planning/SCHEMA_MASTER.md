# Dell-AITC Neo4j Schema Master Document

## Overview
This document serves as the authoritative source for the Dell-AITC Neo4j database schema. It defines all data models, relationships, constraints, and extensibility points.

## 1. AI Technology Categories Domain

### Node: AICategory
The foundation of our classification system, representing distinct AI technology areas.
```yaml
Properties:
  id: string                  # Unique identifier (required)
  name: string               # Category name (required)
  category_definition: string # Detailed description (required)
  status: string            # Active, Deprecated, etc. (required)
  maturity_level: string    # Emerging, Established, etc. (required)
  zone: string              # Technical zone classification (required)
  created_at: datetime      # Creation timestamp (required)
  last_updated: datetime    # Last modification timestamp (required)

Relationships:
  BELONGS_TO: Zone          # Technical zone categorization (required, 1:1)
  TAGGED_WITH: Keyword      # Associated keywords/terms (optional, 1:many)
  HAS_CAPABILITY: Capability # Core capabilities (optional, 1:many)
  DEPENDS_ON: AICategory    # Inter-category dependencies (optional, 1:many)
  INTEGRATES_VIA: IntegrationPattern # Integration methods (optional, 1:many)
  CLASSIFIES: AIClassification # Classification relationships (optional, 1:many)
```

### Node: Zone
Technical zones for categorizing AI technologies.
```yaml
Properties:
  id: string               # Unique identifier (required)
  name: string            # Zone name (required)
  description: string     # Zone description (required)
  created_at: datetime    # Creation timestamp (required)
  last_updated: datetime  # Last modification timestamp (required)

Relationships:
  BELONGS_TO: AICategory (incoming) # Categories in this zone (1:many)
```

### Node: Keyword
Keywords and terms associated with AI categories.
```yaml
Properties:
  id: string              # Unique identifier (required)
  name: string           # Keyword term (required, unique)
  relevance_score: float # Term importance (optional)
  source: string         # Term origin (optional)

Relationships:
  TAGGED_WITH: AICategory (incoming) # Categories using this keyword (many:1)
```

### Node: Capability
Core capabilities of AI categories.
```yaml
Properties:
  id: string              # Unique identifier (required)
  name: string           # Capability name (required)
  description: string    # Detailed description (required)
  type: string          # Capability type (required)

Relationships:
  HAS_CAPABILITY: AICategory (incoming) # Categories with this capability (many:1)
```

### Node: IntegrationPattern
Integration patterns for AI technologies.
```yaml
Properties:
  id: string              # Unique identifier (required)
  name: string           # Pattern name (required)
  description: string    # Pattern description (required)
  best_practices: string # Implementation guidance (optional)

Relationships:
  INTEGRATES_VIA: AICategory (incoming) # Categories using this pattern (many:1)
```

## 2. Federal Use Cases Domain

### Node: UseCase
Represents federal agency AI implementations.
```yaml
Properties:
  id: string                # Unique identifier (required)
  name: string             # Use case name (required)
  agency: string           # Implementing agency (required)
  bureau: string           # Implementing bureau (optional)
  topic_area: string       # Primary topic area (required)
  purpose_benefits: string # Intended benefits (required)
  outputs: string         # Expected outputs (required)
  dev_stage: string       # Development stage (required)
  infrastructure: string  # Technical infrastructure (optional)
  has_ato: boolean       # Authority to Operate status (required)
  contains_pii: boolean  # PII handling flag (required)
  date_initiated: date   # Start date (required)
  updated_at: datetime   # Last update timestamp (required)
  description: string    # Detailed description (required)
  status: string        # Active, Planned, Completed, etc. (required)

Relationships:
  HAS_PURPOSE: PurposeBenefit # Intended benefits (required, 1:many)
  PRODUCES: Output           # Generated outputs (required, 1:many)
  USES_SYSTEM: System        # Associated systems (optional, 1:many)
  CLASSIFIED_AS: AIClassification # AI technology classifications (optional, 1:many)
  BELONGS_TO: Agency         # Implementing agency (required, many:1)
  HAS_ANALYSIS: UnmatchedAnalysis # Analysis when no category matches (optional, 1:many)
```

### Node: Agency
Federal agencies implementing AI use cases.
```yaml
Properties:
  id: string              # Unique identifier (required)
  name: string           # Full agency name (required)
  abbreviation: string  # Agency abbreviation (required, unique)
  type: string         # Agency type (required)
  sector: string       # Government sector (required)

Relationships:
  HAS_BUREAU: Bureau    # Sub-organizations (optional, 1:many)
  HAS_USE_CASE: UseCase # Agency use cases (optional, 1:many)
```

### Node: Bureau
Sub-organizations within federal agencies.
```yaml
Properties:
  id: string              # Unique identifier (required)
  name: string           # Bureau name (required)
  abbreviation: string  # Bureau abbreviation (optional)
  parent_agency: string # Parent agency ID (required)

Relationships:
  BELONGS_TO: Agency (incoming) # Parent agency (required, many:1)
```

### Node: PurposeBenefit
Intended benefits and purposes of use cases.
```yaml
Properties:
  id: string              # Unique identifier (required)
  description: string    # Benefit description (required)
  category: string      # Benefit category (required)
  impact_level: string  # Impact assessment (optional)

Relationships:
  HAS_PURPOSE: UseCase (incoming) # Associated use cases (many:1)
```

### Node: Output
Outputs and deliverables of use cases.
```yaml
Properties:
  id: string              # Unique identifier (required)
  description: string    # Output description (required)
  type: string          # Output type (required)
  format: string        # Output format (optional)

Relationships:
  PRODUCES: UseCase (incoming) # Source use cases (many:1)
```

### Node: System
Technical systems used in implementations.
```yaml
Properties:
  id: string              # Unique identifier (required)
  name: string           # System name (required)
  type: string          # System type (required)
  description: string   # System description (required)
  status: string       # Operational status (required)

Relationships:
  USES_SYSTEM: UseCase (incoming) # Use cases using this system (many:1)
```

## 3. Classification Domain

### Node: AIClassification
Represents the relationship between AI categories and use cases.
```yaml
Properties:
  id: string                # Unique identifier (required)
  type: string             # PRIMARY, SECONDARY, RELATED (required)
  confidence_score: float  # AI-generated confidence score (required)
  justification: string    # Explanation of classification (required)
  classified_at: datetime  # Classification timestamp (required)
  classified_by: string    # AI model or human reviewer (required)
  status: string          # PROPOSED, REVIEWED, APPROVED, REJECTED (required)
  review_notes: string    # Notes from human review (optional)
  method_scores: json     # Individual method scores (required)
  version: string         # Classification model version (required)

Relationships:
  CLASSIFIES: AICategory (incoming) # Classifying category (required, many:1)
  CLASSIFIED_AS: UseCase (incoming) # Classified use case (required, many:1)
```

## 4. Unmatched Analysis Domain

### Node: UnmatchedAnalysis
Analysis of use cases that don't match existing categories.
```yaml
Properties:
  id: string                # Unique identifier (required)
  timestamp: datetime      # Analysis timestamp (required)
  reason_category: string  # NOVEL_TECH, IMPLEMENTATION_SPECIFIC, NON_AI, UNCLEAR_DESC, OTHER (required)
  llm_analysis: string     # Full LLM analysis text (required)
  improvement_suggestions: string # Suggested improvements (required)
  potential_categories: array # Potential new categories to consider (optional)
  best_scores: json        # Scores from each method (required)
  analyzed_at: datetime    # Analysis timestamp (required)
  analyzed_by: string      # Analyzing model/system (required)
  status: string          # NEW, REVIEWED, ACTIONED (required)
  review_notes: string    # Expert review notes (optional)
  action_taken: string    # Actions taken based on analysis (optional)
  resolution_date: datetime # When/if resolved (optional)

Relationships:
  HAS_ANALYSIS: UseCase (incoming) # Analyzed use case (required, many:1)
  SUGGESTS_NEW_CATEGORY: AICategory # Suggested new categories (optional, many:many)
```

## 5. Schema Metadata

### Node: SchemaMetadata
```yaml
Properties:
  id: string                # Unique identifier (required)
  version: string          # Schema version (required)
  supported_investment_stages: array # Valid investment stages (required)
  supported_agency_types: array # Valid agency types (required)
  supported_use_case_statuses: array # Valid statuses (required)
  supported_match_methods: array # Valid matching methods (required)
  supported_relationship_types: array # Valid relationships (required)
  last_updated: datetime   # Last update timestamp (required)

Relationships:
  CURRENT_VERSION: Version # Current schema version (required, 1:1)
```

### Node: Version
```yaml
Properties:
  id: string              # Unique identifier (required)
  number: string         # Version number (required)
  author: string        # Version author (required)
  created_at: datetime  # Creation timestamp (required)
  changes: string      # Change description (required)
  migration_script: string # Schema migration script (optional)

Relationships:
  CURRENT_VERSION: SchemaMetadata (incoming) # Schema using this version (1:1)
```

## Constraints and Indexes

### Uniqueness Constraints
```cypher
// Core Domain Constraints
CREATE CONSTRAINT unique_category_id IF NOT EXISTS
FOR (c:AICategory) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT unique_zone_id IF NOT EXISTS
FOR (z:Zone) REQUIRE z.id IS UNIQUE;

CREATE CONSTRAINT unique_keyword_name IF NOT EXISTS
FOR (k:Keyword) REQUIRE k.name IS UNIQUE;

// Use Case Domain Constraints
CREATE CONSTRAINT unique_usecase_id IF NOT EXISTS
FOR (u:UseCase) REQUIRE u.id IS UNIQUE;

CREATE CONSTRAINT unique_agency_abbrev IF NOT EXISTS
FOR (a:Agency) REQUIRE a.abbreviation IS UNIQUE;

CREATE CONSTRAINT unique_bureau_id IF NOT EXISTS
FOR (b:Bureau) REQUIRE b.id IS UNIQUE;

// Classification Domain Constraints
CREATE CONSTRAINT unique_classification_id IF NOT EXISTS
FOR (c:AIClassification) REQUIRE c.id IS UNIQUE;

// Unmatched Analysis Domain Constraints
CREATE CONSTRAINT unique_unmatched_analysis_id IF NOT EXISTS
FOR (u:UnmatchedAnalysis) REQUIRE u.id IS UNIQUE;
```

### Indexes
```cypher
// Core Domain Indexes
CREATE INDEX category_name IF NOT EXISTS FOR (c:AICategory) ON (c.name);
CREATE INDEX zone_name IF NOT EXISTS FOR (z:Zone) ON (z.name);
CREATE INDEX keyword_name IF NOT EXISTS FOR (k:Keyword) ON (k.name);

// Use Case Domain Indexes
CREATE INDEX usecase_name IF NOT EXISTS FOR (u:UseCase) ON (u.name);
CREATE INDEX agency_name IF NOT EXISTS FOR (a:Agency) ON (a.name);
CREATE INDEX bureau_name IF NOT EXISTS FOR (b:Bureau) ON (b.name);

// Classification Domain Indexes
CREATE INDEX classification_status IF NOT EXISTS 
FOR (c:AIClassification) ON (c.status);
CREATE INDEX classification_type IF NOT EXISTS 
FOR (c:AIClassification) ON (c.type);

// Unmatched Analysis Domain Indexes
CREATE INDEX unmatched_reason IF NOT EXISTS 
FOR (u:UnmatchedAnalysis) ON (u.reason_category);
CREATE INDEX unmatched_status IF NOT EXISTS 
FOR (u:UnmatchedAnalysis) ON (u.status);
```

## Data Quality Rules

1. **Required Properties**
   - All nodes must have required properties filled
   - Dates must be in ISO format
   - Enums must match predefined values
   - Classification types must be one of: PRIMARY, SECONDARY, RELATED
   - Classification status must be one of: PROPOSED, REVIEWED, APPROVED, REJECTED
   - Unmatched reason must be one of: NOVEL_TECH, IMPLEMENTATION_SPECIFIC, NON_AI, UNCLEAR_DESC, OTHER

2. **Relationship Rules**
   - No orphaned nodes (except root nodes)
   - Circular dependencies must be documented
   - Cross-zone relationships must be flagged
   - Each use case must have at least one PRIMARY classification or one UnmatchedAnalysis
   - SECONDARY and RELATED classifications are optional

3. **Classification Rules**
   - Confidence scores must be between 0 and 1
   - Justification is required for all classifications
   - Review notes required for REVIEWED status
   - Classified_by must reference valid AI model or reviewer
   - Classification dates must be tracked

4. **Naming Conventions**
   - Node labels: PascalCase
   - Relationship types: UPPER_SNAKE_CASE
   - Properties: camelCase
   - IDs: lowercase-with-hyphens

## Schema Evolution Guidelines

1. **Backward Compatibility**
   - Maintain support for existing queries
   - Version significant changes
   - Provide migration paths

2. **Extension Process**
   - Document proposed changes
   - Impact analysis required
   - Test with sample data
   - Migration script required

3. **Deprecation Process**
   - Mark as deprecated
   - Minimum 2 version support
   - Migration path documented 