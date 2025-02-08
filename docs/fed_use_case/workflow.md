# Federal Use Case AI Technology Classification Workflow

This document outlines the process flow for classifying federal AI use cases against technology categories using our suite of scripts.

## Overview

The system consists of three main scripts that work together to classify, validate, and import federal AI use case data:

1. `fed_use_case_ai_classifier.py` - Main classification script
2. `fed_use_case_ai_classifier_validation.py` - Validation script
3. `fed_use_case_ai_importer.py` - Neo4j database import script

## Process Flow

### 1. Classification (fed_use_case_ai_classifier.py)

**Purpose**: Evaluates federal AI use cases against 14 AI technology categories using multiple analysis methods.

**Input**:
- Federal use case data from source files
- Environment variables for API keys and configurations

**Output Files**:
- `fed_use_case_ai_classification_neo4j_[timestamp].csv` - Detailed results including:
  - Use case details (ID, name, agency, bureau)
  - Technology category matches
  - Confidence scores
  - Match method results
  - Relationship types
  
- `fed_use_case_ai_classification_preview_[timestamp].csv` - Simplified view containing:
  - Essential use case information
  - Primary technology category matches
  - Top confidence scores

**Location**: `data/output/results/`

### 2. Validation (fed_use_case_ai_classifier_validation.py)

**Purpose**: Validates classification results and performs quality checks.

**Input**:
- Classification output files from step 1
- Test datasets and validation rules

**Output Files**:
- `validation_report_[timestamp].csv` - Detailed validation results
- `validation_metrics_[timestamp].json` - Quality metrics and statistics
- Test logs in `data/output/logs/`

**Location**: `data/output/test_results/`

### 3. Import (fed_use_case_ai_importer.py)

**Purpose**: Imports validated classification results into Neo4j database.

**Input**:
- Validated classification file: `fed_use_case_ai_classification_neo4j_[timestamp].csv`
- Neo4j database credentials

**Output**:
- Neo4j database updates
- Import logs in `data/output/logs/`

## Running the Workflow

1. **Classification**:
   ```bash
   python fed_use_case_ai_classifier.py
   ```

2. **Validation**:
   ```bash
   python fed_use_case_ai_classifier_validation.py
   ```

3. **Import**:
   ```bash
   python fed_use_case_ai_importer.py
   ```

## Environment Setup

Required environment variables:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=your_key  # Optional - used as fallback
```

## Data Flow Diagram

```
[Source Data] → [Classification] → [Validation] → [Import] → [Neo4j Database]
     ↓              ↓                   ↓             ↓
  Input Files → Detailed CSV →  Validation Report → Import Logs
                Preview CSV     Quality Metrics
```

## Best Practices

1. Always run the validation script before importing to Neo4j
2. Review the preview CSV for quick verification
3. Check validation reports for any quality issues
4. Monitor import logs for successful database updates
5. Maintain consistent naming conventions for output files
6. Archive previous results before new runs

## Troubleshooting

Common issues and solutions:
- Missing environment variables
- Invalid input data format
- Neo4j connection issues
- API rate limiting

For detailed error messages and solutions, refer to `docs/troubleshooting.md`. 