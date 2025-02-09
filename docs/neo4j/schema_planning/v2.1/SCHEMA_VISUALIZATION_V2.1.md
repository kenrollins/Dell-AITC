# Dell-AITC Schema Visualization v2.1

## Overview
This document provides a visual representation of the Dell-AITC schema version 2.1, focusing on the relationships between AI technology categories and federal use cases.

## Core Domain Structure

```mermaid
erDiagram
    %% AI Technology Domain
    AICategory {
        uuid id PK
        string name UK
        text category_definition
        enum status
        enum maturity_level
        uuid zone_id FK
        datetime created_at
        datetime last_updated
        string version
    }

    Zone {
        uuid id PK
        string name UK
        string description
        datetime created_at
        datetime last_updated
    }

    Keyword {
        uuid id PK
        string name UK
        string type
        float relevance_score
        datetime created_at
        datetime last_updated
    }

    %% Federal Use Cases Domain
    UseCase {
        uuid id PK
        string name
        enum topic_area
        enum stage
        enum impact_type
        text purpose_benefits
        text outputs
        enum dev_method
        boolean contains_pii
        boolean has_ato
        string system_name
        datetime date_initiated
        datetime date_acquisition
        datetime date_implemented
        datetime date_retired
        datetime created_at
        datetime last_updated
    }

    Agency {
        uuid id PK
        string name UK
        string abbreviation UK
        datetime created_at
        datetime last_updated
    }

    Bureau {
        uuid id PK
        string name
        uuid agency_id FK
        datetime created_at
        datetime last_updated
    }

    %% Core Relationships
    AICategory ||--o{ UseCase : USES_TECHNOLOGY
    Agency ||--|{ Bureau : HAS_BUREAU
    Agency ||--|{ UseCase : IMPLEMENTED_BY
    Bureau ||--|{ UseCase : MANAGED_BY
    Zone ||--|{ AICategory : BELONGS_TO
    Keyword }o--|| AICategory : HAS_KEYWORD
    AICategory }o--|| AICategory : DEPENDS_ON
```

## Relationship Properties

```mermaid
classDiagram
    class USES_TECHNOLOGY {
        +float confidence_score
        +enum match_method
        +datetime created_at
        +datetime last_updated
        +boolean validated
        +string validation_notes
    }

    class BELONGS_TO {
        +datetime created_at
        +float weight
    }

    class HAS_KEYWORD {
        +float relevance
        +datetime created_at
    }

    class DEPENDS_ON {
        +float strength
        +datetime created_at
    }
```

## Classification Flow

```mermaid
sequenceDiagram
    participant UC as UseCase
    participant M as Matcher
    participant AC as AICategory
    participant K as Keywords
    
    UC->>M: Extract text content
    M->>K: Extract keywords
    K->>AC: Match categories
    AC->>M: Calculate confidence
    M->>UC: Create USES_TECHNOLOGY
    Note over M,UC: Includes confidence score<br/>and match method
```

## Data Flow

```mermaid
graph TD
    subgraph Input
        FD[Federal Data]
        KW[Keywords]
        CAT[Categories]
    end

    subgraph Processing
        PM[Pattern Matching]
        SA[Semantic Analysis]
        CS[Confidence Scoring]
    end

    subgraph Output
        UC[Use Cases]
        TC[Tech Categories]
        REL[Relationships]
    end

    FD --> PM
    KW --> PM
    CAT --> PM
    PM --> SA
    SA --> CS
    CS --> REL
    UC --> REL
    TC --> REL
```

## Legend

### Node Properties
- PK: Primary Key
- UK: Unique Key
- FK: Foreign Key

### Relationship Types
- `||--||` : One-to-one
- `||--|{` : One-to-many (required)
- `||--o{` : One-to-many (optional)
- `}|--||` : Many-to-one (required)
- `}o--||` : Many-to-one (optional)
- `}o--o{` : Many-to-many (optional)

### Property Types
- uuid: Unique identifier
- string: Text value
- text: Long text content
- enum: Enumerated value
- boolean: True/False
- float: Decimal number
- datetime: Timestamp
- array: List of values
- json: Complex data structure 