# Changelog

All notable changes to the Dell-AITC project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New codebase management tools:
  - `identify_deprecated_candidates.py`: Identifies potential candidates for deprecation
  - `manage_deprecated.py`: Moves files to deprecated directory and updates documentation
  - `find_deprecated.sh`: Shell wrapper for identify_deprecated_candidates.py
  - `deprecate.sh`: Shell wrapper for manage_deprecated.py
- Comprehensive documentation in `scripts/README.md`
- Detailed documentation of deprecated scripts in `deprecated/scripts/README.md`

### Changed
- Reorganized scripts directory for better maintainability
- Updated shell scripts to work without conda activation
- Improved documentation with usage examples

### Removed
- Moved 20+ deprecated scripts to `deprecated/scripts` directory:
  - Test scripts: test_llm_analyzer.py, test_openai_connection.py, test_keyword_classifier.py, etc.
  - Utility scripts: check_no_matches.py, debug_tva_data.py, list_agency_abbreviations.py, etc.
  - Shell scripts: run_batch_classification.ps1, run_hybrid_classification.py, etc.
  - Backup/duplicate files: partner_analysis.py.bak, ai_tech_classifier.py.duplicate, etc.

## [1.0.0] - 2025-02-23

### Added
- Initial release of Dell-AITC
- Neo4j schema v2.2
- FastAPI backend
- Next.js frontend
- AI classification system
- Federal use case analysis 