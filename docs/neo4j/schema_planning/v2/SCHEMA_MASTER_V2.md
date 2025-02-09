# Dell-AITC Neo4j Schema Master Document - V2

## Version Information
```yaml
Version: 2.0.0
Last Updated: February 2024
Changes:
  - Simplified keyword structure to three types
  - Enhanced domain separation
  - Formalized zone relationships
  - Added validation rules
  - Updated for 2024 AI inventory support
```

## Domain Overview

The schema is organized into five distinct domains:
1. AI Technology Categories
2. Federal Use Cases
3. Classification
4. Unmatched Analysis
5. Schema Metadata

## 1. AI Technology Categories Domain

### Node: AICategory
Primary node representing distinct AI technology areas.
```yaml
Properties:
  id: string                  # Unique identifier (required)
  name: string               # Category name (required)
  category_definition: string # Detailed description (required)
  status: string            # Active, Deprecated (required)
  maturity_level: string    # Emerging, Established, Mature (required)
  zone_id: string          # Technical zone reference (required)
  created_at: datetime      # Creation timestamp (required)
  last_updated: datetime    # Last modification timestamp (required)
  version: string          # Category version (required)

Relationships:
  BELONGS_TO: Zone          # Technical zone categorization (required, 1:1)
  HAS_KEYWORD: Keyword      # Associated keywords (required, 1:many)
  HAS_CAPABILITY: Capability # Core capabilities (required, 1:many)
  DEPENDS_ON: AICategory    # Inter-category dependencies (optional, many:many)
  CLASSIFIES: AIClassification # Classification relationships (optional, 1:many)
```

### Node: Zone
Technical zones for categorizing AI technologies.
```yaml
Properties:
  id: string               # Unique identifier (required)
  name: string            # Zone name (required)
  description: string     # Zone description (required)
  status: string         # Active, Deprecated (required)
  created_at: datetime    # Creation timestamp (required)
  last_updated: datetime  # Last modification timestamp (required)

Relationships:
  CONTAINS: AICategory    # Categories in this zone (1:many)
```

### Node: Keyword
Simplified keyword structure with three types from original data sources.
```yaml
Properties:
  id: string              # Unique identifier (required)
  name: string           # Keyword term (required, unique)
  type: string          # Enum: technical_keywords | capabilities | business_language (required)
  source_column: string  # Original CSV column name for tracking (optional)
  relevance_score: float # Term importance (required)
  status: string        # Active, Deprecated (required)
  created_at: datetime   # Creation timestamp (required)
  last_updated: datetime # Last modification timestamp (required)

Relationships:
  DESCRIBES: AICategory  # Categories using this keyword (many:many)
```

### Node: Capability
Core capabilities of AI categories.
```yaml
Properties:
  id: string              # Unique identifier (required)
  name: string           # Capability name (required)
  description: string    # Detailed description (required)
  type: string          # Capability type (required)
  status: string        # Active, Deprecated (required)
  created_at: datetime   # Creation timestamp (required)
  last_updated: datetime # Last modification timestamp (required)

Relationships:
  ENABLES: AICategory    # Categories with this capability (many:many)
```

## 2. Federal Use Cases Domain

### Node: UseCase
Represents federal agency AI implementations.
```yaml
Properties:
  id: string                # Unique identifier (required)
  name: string             # Use case name (required)
  agency_id: string        # Implementing agency reference (required)
  bureau_id: string        # Implementing bureau reference (optional)
  topic_area: string       # Primary topic area (required)
  description: string      # Detailed description (required)
  purpose: string          # Intended purpose (required)
  benefits: array[string]  # Expected benefits (required)
  dev_stage: string       # Development stage (required)
  infrastructure: string  # Technical infrastructure (optional)
  has_ato: boolean       # Authority to Operate status (required)
  contains_pii: boolean  # PII handling flag (required)
  date_initiated: date   # Start date (required)
  status: string        # Active, Planned, Completed (required)
  created_at: datetime   # Creation timestamp (required)
  last_updated: datetime # Last modification timestamp (required)

Relationships:
  BELONGS_TO: Agency     # Implementing agency (required, many:1)
  PART_OF: Bureau       # Implementing bureau (optional, many:1)
  HAS_OUTCOME: Outcome  # Expected outcomes (required, 1:many)
  USES: System         # Associated systems (optional, 1:many)
  CLASSIFIED_AS: AIClassification # AI classifications (required, 1:many)
  HAS_ANALYSIS: UnmatchedAnalysis # When no category matches (optional, 1:1)
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
  status: string       # Active, Inactive (required)
  created_at: datetime  # Creation timestamp (required)
  last_updated: datetime # Last modification timestamp (required)

Relationships:
  HAS_BUREAU: Bureau    # Sub-organizations (optional, 1:many)
  IMPLEMENTS: UseCase   # Agency use cases (optional, 1:many)
```

### Node: Bureau
Sub-organizations within federal agencies.
```yaml
Properties:
  id: string              # Unique identifier (required)
  name: string           # Bureau name (required)
  abbreviation: string  # Bureau abbreviation (optional)
  agency_id: string     # Parent agency reference (required)
  status: string       # Active, Inactive (required)
  created_at: datetime  # Creation timestamp (required)
  last_updated: datetime # Last modification timestamp (required)

Relationships:
  BELONGS_TO: Agency    # Parent agency (required, many:1)
  IMPLEMENTS: UseCase   # Bureau use cases (optional, 1:many)
```

### Node: Outcome
Expected outcomes of use cases.
```yaml
Properties:
  id: string              # Unique identifier (required)
  description: string    # Outcome description (required)
  type: string          # Outcome type (required)
  metrics: array[string] # Success metrics (optional)
  status: string        # Planned, Achieved, NotAchieved (required)
  created_at: datetime   # Creation timestamp (required)
  last_updated: datetime # Last modification timestamp (required)

Relationships:
  ACHIEVED_BY: UseCase  # Source use cases (many:many)
```

### Node: System
Technical systems used in implementations.
```yaml
Properties:
  id: string              # Unique identifier (required)
  name: string           # System name (required)
  type: string          # System type (required)
  description: string   # System description (required)
  status: string       # Active, Inactive (required)
  created_at: datetime  # Creation timestamp (required)
  last_updated: datetime # Last modification timestamp (required)

Relationships:
  SUPPORTS: UseCase     # Use cases using this system (many:many)
```

## 3. Classification Domain

### Node: AIClassification
Represents the relationship between AI categories and use cases.
```yaml
Properties:
  id: string                # Unique identifier (required)
  type: string             # PRIMARY, SECONDARY (required)
  confidence_score: float  # AI-generated confidence score (required)
  justification: string    # Explanation of classification (required)
  classified_at: datetime  # Classification timestamp (required)
  classified_by: string    # AI model or human reviewer (required)
  status: string          # PROPOSED, REVIEWED, APPROVED (required)
  review_notes: string    # Notes from human review (optional)
  method_scores: json     # Individual method scores (required)
  version: string         # Classification model version (required)
  created_at: datetime    # Creation timestamp (required)
  last_updated: datetime  # Last modification timestamp (required)

Relationships:
  LINKS: AICategory      # Classifying category (required, many:1)
  DESCRIBES: UseCase     # Classified use case (required, many:1)
```

## 4. Unmatched Analysis Domain

### Node: UnmatchedAnalysis
Analysis of use cases that don't match existing categories.
```yaml
Properties:
  id: string                # Unique identifier (required)
  reason: string           # NOVEL_TECH, NON_AI, UNCLEAR (required)
  analysis: string         # Full analysis text (required)
  suggestions: string      # Improvement suggestions (required)
  new_categories: array    # Potential new categories (optional)
  confidence_scores: json  # Method scores (required)
  analyzed_at: datetime    # Analysis timestamp (required)
  analyzed_by: string      # Analyzing model/system (required)
  status: string          # NEW, REVIEWED, ACTIONED (required)
  review_notes: string    # Expert review notes (optional)
  resolution: string      # Resolution details (optional)
  created_at: datetime    # Creation timestamp (required)
  last_updated: datetime  # Last modification timestamp (required)

Relationships:
  ANALYZES: UseCase      # Analyzed use case (required, many:1)
  SUGGESTS: AICategory   # Suggested new categories (optional, many:many)
```

## 5. Schema Metadata

### Node: SchemaVersion
```yaml
Properties:
  id: string                # Unique identifier (required)
  version: string          # Schema version (required)
  valid_stages: array      # Valid development stages (required)
  valid_statuses: array    # Valid status values (required)
  valid_types: array       # Valid classification types (required)
  created_at: datetime     # Creation timestamp (required)
  last_updated: datetime   # Last modification timestamp (required)

Relationships:
  CURRENT: Version        # Current schema version (required, 1:1)
```

## Constraints and Indexes

### Uniqueness Constraints
```cypher
// Core Domain
CREATE CONSTRAINT unique_category_id IF NOT EXISTS
FOR (c:AICategory) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT unique_zone_id IF NOT EXISTS
FOR (z:Zone) REQUIRE z.id IS UNIQUE;

CREATE CONSTRAINT unique_keyword_name IF NOT EXISTS
FOR (k:Keyword) REQUIRE (k.name, k.type) IS UNIQUE;

CREATE CONSTRAINT keyword_type_values IF NOT EXISTS
FOR (k:Keyword) 
REQUIRE k.type IN ['technical_keywords', 'capabilities', 'business_language'];

// Use Case Domain
CREATE CONSTRAINT unique_usecase_id IF NOT EXISTS
FOR (u:UseCase) REQUIRE u.id IS UNIQUE;

CREATE CONSTRAINT unique_agency_abbrev IF NOT EXISTS
FOR (a:Agency) REQUIRE a.abbreviation IS UNIQUE;

// Classification Domain
CREATE CONSTRAINT unique_classification_id IF NOT EXISTS
FOR (c:AIClassification) REQUIRE c.id IS UNIQUE;
```

### Indexes
```cypher
// Core Domain
CREATE INDEX category_name IF NOT EXISTS FOR (c:AICategory) ON (c.name);
CREATE INDEX zone_name IF NOT EXISTS FOR (z:Zone) ON (z.name);
CREATE INDEX keyword_name_type IF NOT EXISTS FOR (k:Keyword) ON (k.name, k.type);

// Use Case Domain
CREATE INDEX usecase_name IF NOT EXISTS FOR (u:UseCase) ON (u.name);
CREATE INDEX agency_name IF NOT EXISTS FOR (a:Agency) ON (a.name);

// Classification Domain
CREATE INDEX classification_status IF NOT EXISTS 
FOR (c:AIClassification) ON (c.status, c.type);
```

## Data Quality Rules

### 1. Required Properties
```yaml
All Nodes:
  - Must have id, created_at, last_updated
  - Dates in ISO format
  - Status values from predefined set
  - Version tracking where applicable

Classifications:
  - Type must be PRIMARY or SECONDARY
  - Status must be PROPOSED, REVIEWED, or APPROVED
  - Confidence scores between 0 and 1
```

### 2. Relationship Rules
```yaml
Cardinality:
  - UseCase must have exactly one PRIMARY classification
  - SECONDARY classifications are optional
  - Each UseCase must have either classification or analysis
  - No orphaned nodes except root nodes

Validation:
  - Cross-domain relationships must be documented
  - Circular dependencies must be justified
  - All relationships must maintain referential integrity
```

### 3. Domain Rules
```yaml
AI Categories:
  - Must belong to exactly one Zone
  - Must have at least one Keyword of each type:
    * technical_keywords
    * capabilities
    * business_language
  - Must have at least one Capability

Use Cases:
  - Must belong to exactly one Agency
  - Must have at least one Outcome
  - Must have valid classification or analysis

Classification:
  - Must include confidence scores
  - Must include method details
  - Must track classification source
```

## Migration Guidelines

### 1. Version Compatibility
```yaml
Backward Compatibility:
  - Maintain support for v1 queries
  - Provide migration scripts
  - Document breaking changes

Forward Planning:
  - Design for extensibility
  - Support future AI categories
  - Allow for schema evolution
```

### 2. Data Migration
```yaml
Process:
  1. Backup existing data
  2. Create new schema
  3. Transform data
  4. Validate migration
  5. Update applications

Validation:
  - Verify node counts
  - Check relationship integrity
  - Validate constraints
  - Test queries
```

### 3. Application Updates
```yaml
Required Updates:
  - Update data importers
  - Modify classifiers
  - Update visualizations
  - Revise documentation
``` 