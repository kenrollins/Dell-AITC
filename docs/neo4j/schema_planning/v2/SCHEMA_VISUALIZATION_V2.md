# Dell-AITC Schema Visualization V2

## Core Domain Structure

```mermaid
graph TD
    subgraph "AI Technology Categories Domain"
        AC[AICategory]
        Z[Zone]
        K[Keyword]
        C[Capability]
        
        Z -->|CONTAINS| AC
        K -->|DESCRIBES| AC
        C -->|ENABLES| AC
        AC -->|DEPENDS_ON| AC
    end
    
    subgraph "Federal Use Cases Domain"
        UC[UseCase]
        A[Agency]
        B[Bureau]
        O[Outcome]
        S[System]
        
        A -->|HAS_BUREAU| B
        A -->|IMPLEMENTS| UC
        B -->|IMPLEMENTS| UC
        O -->|ACHIEVED_BY| UC
        S -->|SUPPORTS| UC
    end
    
    subgraph "Classification Domain"
        CL[AIClassification]
        
        AC -->|CLASSIFIES| CL
        CL -->|DESCRIBES| UC
    end
    
    subgraph "Unmatched Analysis Domain"
        UA[UnmatchedAnalysis]
        
        UA -->|ANALYZES| UC
        UA -->|SUGGESTS| AC
    end
```

## AI Technology Categories Detail

```mermaid
erDiagram
    AICategory {
        string id PK
        string name
        string category_definition
        string status
        string maturity_level
        string zone_id FK
        datetime created_at
        datetime last_updated
        string version
    }
    
    Zone {
        string id PK
        string name
        string description
        string status
        datetime created_at
        datetime last_updated
    }
    
    Keyword {
        string id PK
        string name
        string type "technical_keywords | capabilities | business_language"
        string source_column "Original CSV column"
        float relevance_score
        string status
        datetime created_at
        datetime last_updated
    }
    
    Capability {
        string id PK
        string name
        string description
        string type
        string status
        datetime created_at
        datetime last_updated
    }
    
    Zone ||--|{ AICategory : CONTAINS
    Keyword }|--|| AICategory : DESCRIBES
    Capability }|--|| AICategory : ENABLES
    AICategory }|--|| AICategory : DEPENDS_ON
```

## Federal Use Cases Detail

```mermaid
erDiagram
    UseCase {
        string id PK
        string name
        string agency_id FK
        string bureau_id FK
        string topic_area
        string description
        string purpose
        array benefits
        string dev_stage
        string infrastructure
        boolean has_ato
        boolean contains_pii
        date date_initiated
        string status
        datetime created_at
        datetime last_updated
    }
    
    Agency {
        string id PK
        string name
        string abbreviation UK
        string type
        string sector
        string status
        datetime created_at
        datetime last_updated
    }
    
    Bureau {
        string id PK
        string name
        string abbreviation
        string agency_id FK
        string status
        datetime created_at
        datetime last_updated
    }
    
    Outcome {
        string id PK
        string description
        string type
        array metrics
        string status
        datetime created_at
        datetime last_updated
    }
    
    System {
        string id PK
        string name
        string type
        string description
        string status
        datetime created_at
        datetime last_updated
    }
    
    Agency ||--|{ Bureau : HAS_BUREAU
    Agency ||--|{ UseCase : IMPLEMENTS
    Bureau ||--|{ UseCase : IMPLEMENTS
    Outcome }|--|| UseCase : ACHIEVED_BY
    System }|--|| UseCase : SUPPORTS
```

## Classification Detail

```mermaid
erDiagram
    AIClassification {
        string id PK
        string type
        float confidence_score
        string justification
        datetime classified_at
        string classified_by
        string status
        string review_notes
        json method_scores
        string version
        datetime created_at
        datetime last_updated
    }
    
    AICategory ||--|{ AIClassification : CLASSIFIES
    UseCase ||--|{ AIClassification : CLASSIFIED_AS
```

## Unmatched Analysis Detail

```mermaid
erDiagram
    UnmatchedAnalysis {
        string id PK
        string reason
        string analysis
        string suggestions
        array new_categories
        json confidence_scores
        datetime analyzed_at
        string analyzed_by
        string status
        string review_notes
        string resolution
        datetime created_at
        datetime last_updated
    }
    
    UseCase ||--o| UnmatchedAnalysis : HAS_ANALYSIS
    UnmatchedAnalysis }o--|| AICategory : SUGGESTS
```

## Schema Version Control

```mermaid
erDiagram
    SchemaVersion {
        string id PK
        string version
        array valid_stages
        array valid_statuses
        array valid_types
        datetime created_at
        datetime last_updated
    }
    
    Version {
        string id PK
        string number
        string changes
        datetime created_at
    }
    
    SchemaVersion ||--|| Version : CURRENT
```

## Legend

### Node Types
- PK: Primary Key
- FK: Foreign Key
- UK: Unique Key

### Relationship Types
- `||--|{` : One-to-many (required)
- `||--o{` : One-to-many (optional)
- `}|--||` : Many-to-one (required)
- `}o--||` : Many-to-one (optional)
- `}|--|{` : Many-to-many (required)
- `}o--o{` : Many-to-many (optional)

### Property Types
- string: Text values
- datetime: ISO format timestamps
- boolean: True/False values
- array: List of values
- json: Complex data structures
- float: Decimal numbers
``` 