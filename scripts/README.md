# Scripts Directory

This directory contains utility scripts for the Dell-AITC project.

## Codebase Management Scripts

| Script | Description | Usage |
|--------|-------------|-------|
| find_deprecated.sh | Identifies potential candidates for deprecation | `./scripts/find_deprecated.sh [directory]` |
| deprecate.sh | Moves files to the deprecated directory and updates README | `./scripts/deprecate.sh "REASON" "REPLACEMENT" FILE1 [FILE2 ...]` |
| identify_deprecated_candidates.py | Python script to identify potential candidates for deprecation | `python scripts/identify_deprecated_candidates.py --directory DIR` |
| manage_deprecated.py | Python script to manage deprecated files | `python scripts/manage_deprecated.py --reason "REASON" --replacement "REPLACEMENT" FILE1 [FILE2 ...]` |

## Classification Scripts

| Script | Description | Usage |
|--------|-------------|-------|
| llm_tech_classifier.py | LLM-based classification of federal use cases | `python scripts/llm_tech_classifier.py [options]` |
| run_classifier.sh | Shell script to run the LLM classifier | `./scripts/run_classifier.sh` |
| analyze_classifications.py | Analyze classification results | `python scripts/analyze_classifications.py <csv_file>` |
| export_classifications.py | Export classifications to CSV | `python scripts/export_classifications.py` |
| run_export.sh | Shell script to run the export and analysis | `./scripts/run_export.sh` |

## Partner Analysis Scripts

| Script | Description | Usage |
|--------|-------------|-------|
| partner_analysis.py | Analyzes partners against AI technology categories | `python scripts/partner_analysis.py [options]` |
| analyze_partner_results.py | Analyzes partner analysis results | `python scripts/analyze_partner_results.py [options]` |
| run_partner_analysis.sh | Shell script to run the partner analysis | `./scripts/run_partner_analysis.sh` |
| run_analysis_report.sh | Shell script to run the analysis report | `./scripts/run_analysis_report.sh [format]` |

## Database Management Scripts

| Script | Description | Usage |
|--------|-------------|-------|
| reset_database.sh | Resets the Neo4j database | `./scripts/reset_database.sh` |
| verify_initialization.py | Verifies database initialization | `python scripts/verify_initialization.py` |
| verify_classifications.py | Verifies classifications in Neo4j | `python scripts/verify_classifications.py <use_case_id>` |
| update_schema.py | Updates the Neo4j schema | `python scripts/update_schema.py` |
| add_missing_index.py | Adds missing indexes to Neo4j | `python scripts/add_missing_index.py` |
| check_index.py | Checks Neo4j indexes | `python scripts/check_index.py` |

## Deprecated Scripts

Scripts that are no longer in use have been moved to the `deprecated/scripts` directory. See the README in that directory for details.

To identify potential candidates for deprecation, run:

```bash
./scripts/find_deprecated.sh
```

To move files to the deprecated directory, run:

```bash
./scripts/deprecate.sh "REASON" "REPLACEMENT" FILE1 [FILE2 ...]
``` 