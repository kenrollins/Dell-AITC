# Neo4j Database Schema Documentation

Generated at: 2025-02-06T21:38:02.323560

## Node Labels

The following node labels are defined in the database:

### :AICategory

Properties:
| Property | Type |
|----------|------|
| id | string |
| status | string |
| maturity_level | string |
| name | string |
| created_at | unknown |
| last_updated | unknown |
| category_definition | string |
| zone | string |

### :Keyword

Properties:
| Property | Type |
|----------|------|
| name | string |

### :Capability

Properties:
| Property | Type |
|----------|------|
| name | string |

### :Zone

Properties:
| Property | Type |
|----------|------|
| name | string |

### :IntegrationPattern

Properties:
| Property | Type |
|----------|------|
| name | string |

### :Version

Properties:
| Property | Type |
|----------|------|
| author | string |
| created_at | unknown |
| number | string |
| changes | string |

### :Metadata

Properties:
| Property | Type |
|----------|------|
| description | string |
| last_updated | unknown |
| schema_version | string |
| version | string |

### :UseCase

Properties:
| Property | Type |
|----------|------|
| has_ato | boolean |
| topic_area | string |
| updated_at | unknown |
| contains_pii | boolean |
| date_initiated | string |
| name | string |
| agency | string |
| purpose_benefits | string |
| outputs | string |
| dev_stage | string |
| infrastructure | string |

### :Agency

Properties:
| Property | Type |
|----------|------|
| name | string |
| abbreviation | string |

### :Bureau

Properties:
| Property | Type |
|----------|------|
| name | string |

### :Output

Properties:
| Property | Type |
|----------|------|
| description | string |

### :PurposeBenefit

Properties:
| Property | Type |
|----------|------|
| description | string |

### :System

Properties:
| Property | Type |
|----------|------|
| name | string |

### :SchemaMetadata

Properties:
| Property | Type |
|----------|------|
| id | string |
| supported_investment_stages | list |
| supported_agency_types | list |
| supported_use_case_statuses | list |
| supported_match_methods | list |
| last_updated | unknown |
| supported_relationship_types | list |
| version | string |

## Relationship Types

The following relationship types are defined in the database:

### :BELONGS_TO

No properties defined.

### :TAGGED_WITH

Properties:
| Property | Type |
|----------|------|
| weight | integer |

### :HAS_CAPABILITY

No properties defined.

### :DEPENDS_ON

Properties:
| Property | Type |
|----------|------|
| created_at | unknown |
| cross_zone | boolean |

### :INTEGRATES_VIA

No properties defined.

### :CURRENT_VERSION

No properties defined.

### :HAS_BUREAU

No properties defined.

### :HAS_USE_CASE

No properties defined.

### :USES_SYSTEM

No properties defined.

### :HAS_PURPOSE

No properties defined.

### :PRODUCES

No properties defined.
