# Dell-AITC Schema Visualization

## Core Schema

```mermaid
erDiagram
    %% Core AI Technology Categories
    AICategory {
        string id
        string name
        string category_definition
        string status
        string maturity_level
        string zone
        datetime created_at
        datetime last_updated
    }

    Zone {
        string name
    }

    Keyword {
        string name
    }

    Capability {
        string name
    }

    IntegrationPattern {
        string name
    }

    %% Federal Use Case Structure
    UseCase {
        string name
        string agency
        string topic_area
        string purpose_benefits
        string outputs
        string dev_stage
        string infrastructure
        boolean has_ato
        boolean contains_pii
        string date_initiated
        datetime updated_at
    }

    %% Classification Relationship
    AIClassification {
        string type              %% PRIMARY, SECONDARY, RELATED
        float confidence_score   %% AI-generated confidence score
        string justification     %% Explanation of classification
        datetime classified_at   %% When the classification was made
        string classified_by     %% AI model or human reviewer
        string status           %% PROPOSED, REVIEWED, APPROVED, REJECTED
        string review_notes     %% Notes from human review
    }

    %% NO_MATCH Analysis Structure
    NoMatchAnalysis {
        string id
        datetime timestamp
        string reason_category     %% NOVEL_TECH, IMPLEMENTATION_SPECIFIC, NON_AI, UNCLEAR_DESC, OTHER
        string llm_analysis        %% Full LLM analysis text
        string improvement_suggestions
        array potential_categories %% Potential new categories to consider
        json best_scores          %% Scores from each method
        datetime analyzed_at
        string analyzed_by
        string status             %% NEW, REVIEWED, ACTIONED
        string review_notes
    }

    Agency {
        string name
        string abbreviation
    }

    Bureau {
        string name
    }

    System {
        string name
    }

    PurposeBenefit {
        string description
    }

    Output {
        string description
    }

    %% Metadata and Versioning
    SchemaMetadata {
        string id
        string version
        array supported_investment_stages
        array supported_agency_types
        array supported_use_case_statuses
        array supported_match_methods
        array supported_relationship_types
        datetime last_updated
    }

    Version {
        string author
        datetime created_at
        string number
        string changes
    }

    %% Core Category Relationships
    AICategory ||--|| Zone : BELONGS_TO
    AICategory ||--o{ Keyword : TAGGED_WITH
    AICategory ||--o{ Capability : HAS_CAPABILITY
    AICategory ||--o{ AICategory : DEPENDS_ON
    AICategory ||--o{ IntegrationPattern : INTEGRATES_VIA

    %% Classification Relationships
    AICategory ||--o{ AIClassification : CLASSIFIES
    UseCase ||--o{ AIClassification : CLASSIFIED_AS
    
    %% NO_MATCH Relationships
    UseCase ||--o{ NoMatchAnalysis : NO_MATCH_ANALYSIS
    NoMatchAnalysis ||--o{ AICategory : SUGGESTS_NEW_CATEGORY
    
    %% Use Case Structure
    Agency ||--|{ UseCase : HAS_USE_CASE
    Agency ||--|{ Bureau : HAS_BUREAU
    UseCase ||--o{ PurposeBenefit : HAS_PURPOSE
    UseCase ||--o{ Output : PRODUCES
    UseCase ||--o{ System : USES_SYSTEM

    %% Metadata Relationships
    SchemaMetadata ||--|| Version : CURRENT_VERSION
```

## NO_MATCH Analysis Flow

```mermaid
graph TD
    UC[UseCase] -->|No Category Match| NMA[NoMatchAnalysis]
    NMA -->|Categorized As| RC[Reason Category]
    RC -->|NOVEL_TECH| NT[New Technology Area]
    RC -->|IMPLEMENTATION_SPECIFIC| IS[Implementation Details]
    RC -->|NON_AI| NA[Non-AI System]
    RC -->|UNCLEAR_DESC| UD[Unclear Description]
    RC -->|OTHER| OT[Other Reasons]
    
    NMA -->|Generates| IS[Improvement Suggestions]
    NMA -->|Suggests| NC[New Categories]
    
    subgraph "Analysis Process"
        LLM[LLM Analysis] -->|Provides| JU[Justification]
        LLM -->|Extracts| KP[Key Points]
        LLM -->|Recommends| AC[Action Items]
    end
    
    subgraph "Review Workflow"
        NMA -->|Status| ST[Status Tracking]
        ST -->|NEW| New[Newly Analyzed]
        ST -->|REVIEWED| Rev[Expert Reviewed]
        ST -->|ACTIONED| Act[Changes Implemented]
    end
```

## Extension Points

```mermaid
erDiagram
    %% Core Entities
    AICategory {
        string id
        string name
    }

    UseCase {
        string name
        string agency
    }

    AIClassification {
        string type
        float confidence_score
        string justification
        datetime classified_at
        string classified_by
        string status
        string review_notes
    }

    %% Dell Solutions
    DellSolution {
        string id
        string name
        string category
        string description
    }

    %% Partner Solutions
    PartnerSolution {
        string id
        string partner_name
        string solution_name
        string certification_status
    }

    %% Evaluation Framework
    Evaluation {
        string id
        datetime date
        float score
        array criteria
    }

    %% Relationships
    DellSolution ||--o{ AICategory : IMPLEMENTS
    DellSolution ||--o{ UseCase : DEPLOYED_IN
    DellSolution ||--o{ Capability : PROVIDES

    PartnerSolution ||--|| DellSolution : COMPLEMENTS
    PartnerSolution ||--o{ AICategory : IMPLEMENTS
    PartnerSolution ||--o{ UseCase : DEPLOYED_IN

    Evaluation ||--o{ DellSolution : EVALUATES
    Evaluation ||--o{ PartnerSolution : EVALUATES
    Evaluation ||--o{ UseCase : APPLIES_TO
    Evaluation ||--o{ AICategory : BASED_ON
```

## Schema Evolution

```mermaid
erDiagram
    %% Version 1.0 - Basic Classification
    AICategory_v1 ||--o{ UseCase_v1 : CLASSIFIES
    
    %% Version 2.0 - Enhanced Classification with Relationship Entity
    AICategory_v2 ||--o{ AIClassification : CLASSIFIES
    UseCase_v2 ||--o{ AIClassification : CLASSIFIED_AS
    AICategory_v2 ||--o{ AICategory_v2 : DEPENDS_ON
    AICategory_v2 ||--o{ Capability : HAS_CAPABILITY
    
    %% Version 3.0 - Dell Integration and NO_MATCH Analysis
    AICategory_v3 ||--o{ AIClassification : CLASSIFIES
    UseCase_v3 ||--o{ AIClassification : CLASSIFIED_AS
    UseCase_v3 ||--o{ NoMatchAnalysis : NO_MATCH_ANALYSIS
    DellSolution ||--o{ AICategory_v3 : IMPLEMENTS
    DellSolution ||--o{ UseCase_v3 : DEPLOYED_IN
    PartnerSolution ||--o{ DellSolution : COMPLEMENTS
```

## Cardinality Guide

- `||--||` : One-to-one relationship
- `||--|{` : One-to-many relationship (required)
- `||--o{` : One-to-many relationship (optional)
- `}|--|{` : Many-to-many relationship

## Color Guide

- Core Entities: Primary business objects
- Supporting Entities: Auxiliary data structures
- Extension Points: Future expansion capabilities
- Metadata: Schema versioning and management 