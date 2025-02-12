# Schema v2.1 Changes

## Overview
This document outlines the planned changes for schema v2.1, designed for clean-slate implementation with 2024 Federal inventory data.

## Key Objectives
1. Optimize for 2024 Federal inventory data structure
2. Improve query performance with better indexing
3. Enhance data validation rules
4. Simplify relationship patterns

## Schema Structure

### Node Types
1. AICategory
   - Core AI technology categories
   - Enhanced validation rules
   - Improved property constraints

2. Zone
   - Technical zones for categorization
   - Clearer boundary definitions
   - Simplified relationships

3. Keyword
   - Three distinct types:
     - technical_keywords
     - capabilities
     - business_language
   - Improved relevance scoring
   - Better categorization rules

4. Capability
   - Enhanced description requirements
   - Clearer type definitions
   - Improved relationship rules

### Required Properties

#### AICategory
```yaml
properties:
  id:
    type: string
    format: uuid
    unique: true
    indexed: true
  name:
    type: string
    unique: true
    indexed: true
  category_definition:
    type: string
    min_length: 50
  status:
    type: string
    enum: [active, deprecated]
  maturity_level:
    type: string
    enum: [emerging, established, mature]
  zone_id:
    type: string
    format: uuid
  created_at:
    type: datetime
  last_updated:
    type: datetime
  version:
    type: string
    pattern: "^\\d+\\.\\d+\\.\\d+$"
```

#### Zone
```yaml
properties:
  id:
    type: string
    format: uuid
    unique: true
    indexed: true
  name:
    type: string
    unique: true
    indexed: true
  description:
    type: string
    min_length: 30
  status:
    type: string
    enum: [active, deprecated]
  created_at:
    type: datetime
  last_updated:
    type: datetime
```

#### Keyword
```yaml
properties:
  id:
    type: string
    format: uuid
    unique: true
    indexed: true
  name:
    type: string
    unique: true
    indexed: true
  type:
    type: string
    enum: [technical_keywords, capabilities, business_language]
  relevance_score:
    type: float
    minimum: 0.0
    maximum: 1.0
  status:
    type: string
    enum: [active, deprecated]
  created_at:
    type: datetime
  last_updated:
    type: datetime
```

### Relationships
```yaml
relationships:
  BELONGS_TO:
    source: AICategory
    target: Zone
    cardinality: many_to_one
    properties:
      created_at: datetime
      weight: float

  HAS_KEYWORD:
    source: AICategory
    target: Keyword
    cardinality: many_to_many
    properties:
      relevance: float
      created_at: datetime

  DEPENDS_ON:
    source: AICategory
    target: AICategory
    cardinality: many_to_many
    properties:
      strength: float
      created_at: datetime
```

## Validation Rules

### Property Validation
1. All UUIDs must follow v4 format
2. All names must be unique within their node type
3. Dates must be in ISO format
4. Version numbers must follow semantic versioning
5. Relevance scores must be between 0 and 1

### Relationship Validation
1. No orphaned nodes allowed
2. Circular dependencies prevented in DEPENDS_ON
3. Each AICategory must have at least one keyword
4. Each AICategory must belong to exactly one Zone

## Indexes and Constraints

### Unique Constraints
```cypher
CREATE CONSTRAINT FOR (c:AICategory) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT FOR (c:AICategory) REQUIRE c.name IS UNIQUE;
CREATE CONSTRAINT FOR (z:Zone) REQUIRE z.id IS UNIQUE;
CREATE CONSTRAINT FOR (z:Zone) REQUIRE z.name IS UNIQUE;
CREATE CONSTRAINT FOR (k:Keyword) REQUIRE k.id IS UNIQUE;
CREATE CONSTRAINT FOR (k:Keyword) REQUIRE k.name IS UNIQUE;
```

### Indexes
```cypher
CREATE INDEX FOR (c:AICategory) ON (c.status);
CREATE INDEX FOR (c:AICategory) ON (c.maturity_level);
CREATE INDEX FOR (k:Keyword) ON (k.type);
CREATE INDEX FOR (k:Keyword) ON (k.status);
```

## Data Quality Rules
1. Category definitions must be comprehensive
2. Keywords must be normalized
3. Relationships must have valid properties
4. All required metadata must be present

## Implementation Steps
1. Create schema JSON file
2. Set up validation rules
3. Create indexes and constraints
4. Implement data loading procedures
5. Set up monitoring

## Next Steps
1. [ ] Review and finalize property definitions
2. [ ] Create schema JSON file
3. [ ] Update schema validator
4. [ ] Create data loading scripts
5. [ ] Set up automated testing

## Schema Changes - February 2024

## Version 2.1.1
Status: Completed
Date: February 2024

### Changes
1. Agency Node Constraint Modification
```yaml
Type: Constraint Modification
Status: Completed
Impact: Low
Node: Agency
Change:
  - Remove unique constraint on Agency.abbreviation
  - Keep index on Agency.abbreviation for query performance
Reason: Multiple agencies may share the same abbreviation in the federal inventory data
Validation:
  - Verified existing queries don't rely on abbreviation uniqueness
  - Confirmed index still supports efficient agency lookups
Migration:
  - Dropped existing unique constraint
  - Maintained regular index for query performance
  - Updated data loading scripts to handle duplicate abbreviations
Documentation Updates:
  - Updated neo4j_schema_v2.1.json
  - Updated SCHEMA_MASTER_V2.1.md
  - Updated SCHEMA_VISUALIZATION_V2.1.md
```

2. Schema Setup Function Enhancement
```yaml
Type: Implementation Enhancement
Status: Completed
Impact: Low
Component: Database Setup
Change:
  - Added drop_constraints_and_indexes function
  - Implemented safe constraint/index cleanup
Reason: Need proper cleanup of existing constraints and indexes before schema setup
Validation:
  - Verified successful cleanup of existing constraints
  - Confirmed proper recreation of new constraints
  - Tested with sample data
Implementation:
  - Added new function to database setup
  - Enhanced error handling and logging
  - Verified cleanup process
```

3. Boolean Field Handling
```yaml
Type: Data Type Enhancement
Status: Completed
Impact: Medium
Node: UseCase
Change:
  - Updated contains_pii and has_ato to be nullable boolean fields
  - Added safe boolean conversion handling
Reason: Inventory data contains null values for boolean fields
Validation:
  - Tested with null values
  - Verified data loading process
  - Confirmed schema compatibility
Implementation:
  - Updated schema to support nullable fields
  - Added safe_to_bool conversion function
  - Enhanced data loading process
```

## Version 2.1.2
Status: Completed
Date: February 2024

### Changes
1. Data Loading Optimization
   - Improved error handling in load_inventory function
   - Added better logging for data loading process
   Reason: Need better visibility into data loading issues
   Implementation:
     - Enhanced error messages
     - Added detailed logging
     - Verified with larger dataset

2. Keyword Node Documentation Fix
   - Added missing Keyword node definition to schema JSON
   - Added Keyword constraints and indexes to setup Cypher
   - Ensured consistent documentation across all schema files
   Reason: Keyword node structure existed in visualization but was missing from core schema docs
   Impact: Documentation Only
   Implementation:
     - Updated neo4j_schema.json
     - Updated setup_schema_v2.1.cypher
     - Added keyword constraints and indexes
     - Verified consistency across all schema documents
   Validation:
     - Confirmed all keyword properties are properly documented
     - Verified index and constraint definitions
     - Checked relationship documentation

## Version 2.1.2 (Next)
Status: Planning
Changes Planned:
- Data Loading Optimization
- Additional validation rules
- Performance enhancements 