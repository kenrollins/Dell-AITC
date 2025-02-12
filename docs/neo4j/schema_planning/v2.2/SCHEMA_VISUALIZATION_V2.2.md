# Dell-AITC Schema Visualization v2.2

## Overview
This document provides a visual representation of the Dell-AITC schema version 2.2, focusing on the relationships between AI technology categories and federal use cases.

## Version Information
```yaml
Version: 2.2
Status: Active
Last Updated: 2024-02
```

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

    %% Classification Domain
    AIClassification {
        uuid id PK
        enum match_type
        integer match_rank
        float confidence
        enum analysis_method
        string analysis_version
        float keyword_score
        float semantic_score
        float llm_score
        json field_match_scores
        json term_match_details
        string[] matched_keywords
        boolean llm_verification
        float llm_confidence
        string llm_reasoning
        json llm_suggestions
        string[] improvement_notes
        boolean false_positive
        boolean manual_override
        enum review_status
        datetime classified_at
        string classified_by
        datetime last_updated
    }

    NoMatchAnalysis {
        uuid id PK
        string reason
        float confidence
        json llm_analysis
        string[] suggested_keywords
        json improvement_suggestions
        datetime created_at
        string analyzed_by
        enum status
        string review_notes
    }

    %% Core Relationships
    AICategory ||--o{ AIClassification : CLASSIFIES
    UseCase ||--o{ AIClassification : CLASSIFIED_AS
    UseCase ||--o{ NoMatchAnalysis : HAS_ANALYSIS
    NoMatchAnalysis }o--|| AICategory : SUGGESTS_CATEGORY
    Agency ||--|{ Bureau : HAS_BUREAU
    Agency ||--|{ UseCase : IMPLEMENTED_BY
    Bureau ||--|{ UseCase : MANAGED_BY
    Zone ||--|{ AICategory : BELONGS_TO
    Keyword }o--|| AICategory : HAS_KEYWORD
    AICategory }o--|| AICategory : DEPENDS_ON
```

## Node Properties

```mermaid
classDiagram
    class AIClassification {
        +enum match_type
        +integer match_rank
        +float confidence
        +enum analysis_method
        +string analysis_version
        +float keyword_score
        +float semantic_score
        +float llm_score
        +json field_match_scores
        +json term_match_details
        +string[] matched_keywords
        +boolean llm_verification
        +float llm_confidence
        +string llm_reasoning
        +json llm_suggestions
        +string[] improvement_notes
        +boolean false_positive
        +boolean manual_override
        +enum review_status
        +datetime classified_at
        +string classified_by
        +datetime last_updated
    }

    class NoMatchAnalysis {
        +string reason
        +float confidence
        +json llm_analysis
        +string[] suggested_keywords
        +json improvement_suggestions
        +datetime created_at
        +string analyzed_by
        +enum status
        +string review_notes
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
    participant CL as AIClassification
    participant K as Keywords
    
    UC->>M: Extract text content
    M->>K: Extract keywords
    K->>AC: Match categories
    AC->>M: Calculate confidence
    M->>CL: Create classification
    CL->>UC: Link to use case
    Note over CL,UC: Includes match type,<br/>confidence scores,<br/>and LLM analysis
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
        LLM[LLM Analysis]
    end

    subgraph Output
        UC[Use Cases]
        TC[Tech Categories]
        CL[Classifications]
        NM[No Matches]
    end

    FD --> PM
    KW --> PM
    CAT --> PM
    PM --> SA
    SA --> CS
    CS --> LLM
    LLM --> CL
    LLM --> NM
    UC --> CL
    TC --> CL
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