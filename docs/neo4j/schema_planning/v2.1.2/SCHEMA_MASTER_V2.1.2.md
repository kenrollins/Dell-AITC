# Dell-AITC Neo4j Schema v2.1.2

## Overview
This schema defines the structure for the Dell-AITC AI Technology Categorization system, focusing on mapping federal AI use cases to technology categories.

## Version Information
```yaml
Version: 2.1.2
Status: Released
Last Updated: 2024-02
Changes from v2.1.1:
  - Updated Agency node constraints
  - Enhanced boolean field handling
  - Improved data loading optimization
```

## Domain Model

### Core Domains
1. AI Technology Classification
   - AI Categories
   - Technical Zones
   - Keywords and Capabilities
2. Federal Use Cases
   - Use Case Details
   - Agency Structure
   - Implementation Status
3. Technology Mapping
   - Category Matching
   - Confidence Scoring
   - Validation Process

## Node Types

### 1. AICategory
Represents distinct AI technology categories.

```yaml
Properties:
  id: uuid                    # Unique identifier
  name: string               # Category name
  category_definition: text  # Detailed description
  status: enum              # active, deprecated
  maturity_level: enum      # emerging, established, mature
  zone_id: uuid            # Technical zone reference
  created_at: datetime     # Creation timestamp
  last_updated: datetime   # Last modification
  version: string          # Semantic version (x.y.z)

Constraints:
  - id must be unique
  - name must be unique
  - category_definition minimum length: 50 chars
  - valid maturity_level values
  - valid status values
```

### 2. UseCase
Represents federal AI use cases from the inventory.

```yaml
Properties:
  id: uuid                 # Unique identifier
  name: string            # Use case name
  topic_area: enum        # Predefined topic areas
  stage: enum            # Development stage
  impact_type: enum      # Rights/Safety impact
  purpose_benefits: text # Description of purpose
  outputs: text         # System outputs
  dev_method: enum      # Development method
  contains_pii: boolean # PII indicator
  has_ato: boolean     # ATO status
  system_name: string   # Optional system name
  date_initiated: datetime    # Start date
  date_acquisition: datetime  # Acquisition date
  date_implemented: datetime  # Implementation date
  date_retired: datetime     # Retirement date
  created_at: datetime      # Record creation
  last_updated: datetime    # Last update

Constraints:
  - id must be unique
  - valid topic_area values
  - valid stage values
  - valid impact_type values
  - valid dev_method values
```

### 3. Agency
Represents federal agencies.

```yaml
Properties:
  id: uuid                # Unique identifier
  name: string           # Full agency name
  abbreviation: string   # Official abbreviation (non-unique)
  created_at: datetime   # Record creation
  last_updated: datetime # Last update

Constraints:
  - id must be unique
  - name must be unique
  - abbreviation is indexed for performance
```

### 4. Bureau
Represents sub-organizations within agencies.

```yaml
Properties:
  id: uuid                # Unique identifier
  name: string           # Bureau name
  agency_id: uuid        # Parent agency reference
  created_at: datetime   # Record creation
  last_updated: datetime # Last update

Constraints:
  - id must be unique
  - valid agency_id reference
```

## Relationships

### 1. USES_TECHNOLOGY
Maps use cases to AI technology categories.

```yaml
Source: UseCase
Target: AICategory
Cardinality: MANY_TO_MANY
Properties:
  confidence_score: float    # Match confidence (0-1)
  match_method: enum        # keyword, semantic, manual, hybrid
  created_at: datetime     # Creation timestamp
  last_updated: datetime   # Last update
  validated: boolean       # Validation status
  validation_notes: string # Optional notes

Constraints:
  - confidence_score between 0 and 1
  - valid match_method values
```

### 2. IMPLEMENTED_BY
Links use cases to implementing agencies.

```yaml
Source: UseCase
Target: Agency
Cardinality: MANY_TO_ONE
Properties:
  created_at: datetime # Creation timestamp
```

### 3. MANAGED_BY
Links use cases to managing bureaus.

```yaml
Source: UseCase
Target: Bureau
Cardinality: MANY_TO_ONE
Properties:
  created_at: datetime # Creation timestamp
```

### 4. BELONGS_TO
Links AI categories to technical zones.

```yaml
Source: AICategory
Target: Zone
Cardinality: MANY_TO_ONE
Properties:
  created_at: datetime # Creation timestamp
  weight: float       # Relationship weight (0-1)
```

### 5. HAS_KEYWORD
Links AI categories to keywords.

```yaml
Source: AICategory
Target: Keyword
Cardinality: MANY_TO_MANY
Properties:
  relevance: float    # Keyword relevance (0-1)
  created_at: datetime # Creation timestamp
```

### 6. DEPENDS_ON
Represents dependencies between AI categories.

```yaml
Source: AICategory
Target: AICategory
Cardinality: MANY_TO_MANY
Properties:
  strength: float     # Dependency strength (0-1)
  created_at: datetime # Creation timestamp
```

## Indexes

### Node Indexes
```cypher
// AICategory
CREATE INDEX ON :AICategory(status)
CREATE INDEX ON :AICategory(maturity_level)

// UseCase
CREATE INDEX ON :UseCase(name)
CREATE INDEX ON :UseCase(topic_area)
CREATE INDEX ON :UseCase(stage)
CREATE INDEX ON :UseCase(impact_type)
CREATE INDEX ON :UseCase(contains_pii)
CREATE INDEX ON :UseCase(has_ato)

// Bureau
CREATE INDEX ON :Bureau(name)
CREATE INDEX ON :Bureau(agency_id)
```

### Fulltext Indexes
```cypher
// Text search indexes
CREATE FULLTEXT INDEX ON :UseCase([purpose_benefits, outputs])
CREATE FULLTEXT INDEX ON :AICategory([category_definition])
```

## Data Quality Rules

### Node Rules
1. AICategory
   - Category definitions must be comprehensive
   - Names must be normalized
   - Version numbers must follow semver
   
2. UseCase
   - Purpose and benefits must be detailed
   - Dates must follow MM/YYYY format
   - Impact type must be explicitly specified
   
3. Agency/Bureau
   - Agency abbreviations must be standardized
   - Bureau names must be consistent

### Relationship Rules
1. USES_TECHNOLOGY
   - Must have confidence score
   - Must specify match method
   - Should be validated for high-impact cases

2. General
   - No orphaned nodes
   - Required timestamps
   - Valid property ranges

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