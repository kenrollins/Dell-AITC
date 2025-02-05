# Neo4j Database Schema Documentation
## Overview
### Node Summary
| Label | Count |
|-------|-------|
| AICategory | 14 |
| Keyword | 237 |
| Capability | 178 |
| Zone | 4 |
| IntegrationPattern | 70 |
| Version | 1 |
| Metadata | 1 |
| UseCase | 2,052 |
| Agency | 42 |
| Bureau | 325 |
| Output | 1,709 |
| PurposeBenefit | 2,094 |
| EvaluationRun | 0 |
| System | 322 |

### Relationship Summary
| Type | Count |
|------|-------|
| BELONGS_TO | 14 |
| TAGGED_WITH | 242 |
| HAS_CAPABILITY | 192 |
| DEPENDS_ON | 24 |
| INTEGRATES_VIA | 84 |
| CURRENT_VERSION | 1 |
| IMPLEMENTS | 0 |
| HAS_BUREAU | 336 |
| HAS_USE_CASE | 2,052 |
| USES_SYSTEM | 768 |
| HAS_PURPOSE | 2,213 |
| PRODUCES | 2,074 |


## Node Labels and Properties

### AICategory
| Property | Type |
|----------|------|
| id | str |
| status | str |
| maturity_level | str |
| name | str |
| created_at | DateTime |
| last_updated | DateTime |
| category_definition | str |
| zone | str |

### Keyword
| Property | Type |
|----------|------|
| name | str |

### Capability
| Property | Type |
|----------|------|
| name | str |

### Zone
| Property | Type |
|----------|------|
| name | str |

### IntegrationPattern
| Property | Type |
|----------|------|
| name | str |

### Version
| Property | Type |
|----------|------|
| author | str |
| created_at | DateTime |
| number | str |
| changes | str |

### Metadata
| Property | Type |
|----------|------|
| description | str |
| last_updated | DateTime |
| schema_version | str |
| version | str |

### UseCase
| Property | Type |
|----------|------|
| has_ato | bool |
| topic_area | str |
| updated_at | DateTime |
| contains_pii | bool |
| date_initiated | str |
| name | str |
| agency | str |
| purpose_benefits | str |
| outputs | str |
| dev_stage | str |
| infrastructure | str |

### Agency
| Property | Type |
|----------|------|
| name | str |
| abbreviation | str |

### Bureau
| Property | Type |
|----------|------|
| name | str |

### Output
| Property | Type |
|----------|------|
| description | str |

### PurposeBenefit
| Property | Type |
|----------|------|
| description | str |

### EvaluationRun
| Property | Type |
|----------|------|

### System
| Property | Type |
|----------|------|
| name | str |


## Relationship Types

### HAS_PURPOSE
Patterns:
- (UseCase) -> (PurposeBenefit) [2,213 instances]

### PRODUCES
Patterns:
- (UseCase) -> (Output) [2,074 instances]

### HAS_USE_CASE
Patterns:
- (Agency) -> (UseCase) [2,052 instances]

### USES_SYSTEM
Patterns:
- (UseCase) -> (System) [768 instances]

### HAS_BUREAU
Patterns:
- (Agency) -> (Bureau) [336 instances]

### TAGGED_WITH
Patterns:
- (AICategory) -> (Keyword) [242 instances]

### HAS_CAPABILITY
Patterns:
- (AICategory) -> (Capability) [192 instances]

### INTEGRATES_VIA
Patterns:
- (AICategory) -> (IntegrationPattern) [84 instances]

### DEPENDS_ON
Patterns:
- (AICategory) -> (AICategory) [24 instances]

### BELONGS_TO
Patterns:
- (AICategory) -> (Zone) [14 instances]

### CURRENT_VERSION
Patterns:
- (Metadata) -> (Version) [1 instances]


## Constraints
| Name | Type | For | Properties |
|------|------|-----|------------|
| agency_name | UNIQUENESS |  | name |
| ai_category_id | UNIQUENESS |  | id |
| bureau_name | UNIQUENESS |  | name |
| capability_name | UNIQUENESS |  | name |
| evaluation_run_id | UNIQUENESS |  | run_id |
| keyword_name | UNIQUENESS |  | name |
| output_composite | UNIQUENESS |  | description, agency |
| purpose_benefit_composite | UNIQUENESS |  | description, agency |
| system_name | UNIQUENESS |  | name |
| use_case_composite | UNIQUENESS |  | name, agency |
| zone_name | UNIQUENESS |  | name |


## Indexes
| Name | Type | For | Properties |
|------|------|-----|------------|
| agency_name | RANGE |  | name |
| ai_category_id | RANGE |  | id |
| ai_category_name | RANGE |  | name |
| ai_category_zone | RANGE |  | zone |
| bureau_name | RANGE |  | name |
| capability_name | RANGE |  | name |
| evaluation_run_id | RANGE |  | run_id |
| evaluation_timestamp | RANGE |  | timestamp |
| index_343aff4e | LOOKUP |  |  |
| index_f7700477 | LOOKUP |  |  |
| keyword_name | RANGE |  | name |
| output_composite | RANGE |  | description, agency |
| purpose_benefit_composite | RANGE |  | description, agency |
| system_name | RANGE |  | name |
| use_case_agency | RANGE |  | agency |
| use_case_composite | RANGE |  | name, agency |
| use_case_confidence | RANGE |  | confidence |
| use_case_match_method | RANGE |  | match_method |
| use_case_match_score | RANGE |  | final_score |
| use_case_topic | RANGE |  | topic_area |
| zone_name | RANGE |  | name |


## Common Patterns
| Source | Relationship | Target | Frequency |
|--------|--------------|--------|------------|
| UseCase | HAS_PURPOSE | PurposeBenefit | 2,213 |
| UseCase | PRODUCES | Output | 2,074 |
| Agency | HAS_USE_CASE | UseCase | 2,052 |
| UseCase | USES_SYSTEM | System | 768 |
| Agency | HAS_BUREAU | Bureau | 336 |
| AICategory | TAGGED_WITH | Keyword | 242 |
| AICategory | HAS_CAPABILITY | Capability | 192 |
| AICategory | INTEGRATES_VIA | IntegrationPattern | 84 |
| AICategory | DEPENDS_ON | AICategory | 24 |
| AICategory | BELONGS_TO | Zone | 14 |
| Metadata | CURRENT_VERSION | Version | 1 |