# Deprecated Scripts

This directory contains scripts that have been deprecated for various reasons. They are kept for reference but should not be used in production.

## Recently Deprecated Scripts

| Script | Deprecated Date | Reason | Replacement |
|--------|----------------|--------|-------------|
| test_keyword_classifier.py | 2025-03-04 | Test script in wrong location | backend/tests/integration |
| setup_keywords.py | 2025-03-04 | Setup script with no active usage | Keyword management in backend/app/services/ai_classifier |
| run_export.sh | 2025-03-04 | Utility script with no active usage | Integrated into run_classifier.sh |
| run_partner_analysis.sh | 2025-03-04 | Utility script with no active usage | Integrated into run_classifier.sh |
| check_keywords.py | 2025-03-04 | Utility script with no active usage | Keyword management in backend/app/services/ai_classifier |
| fix_duplicate_agencies.py | 2025-03-04 | One-time data cleanup utility | Database integrity checks in Neo4j |
| analyze_enums.py | 2025-03-04 | Analysis script with no active usage | Integrated into llm_tech_classifier.py |
| list_agency_abbreviations.py | 2025-03-04 | Utility script with no active usage | Agency information in Neo4j database |
| debug_tva_data.py | 2025-03-04 | One-time debugging utility | llm_tech_classifier.py with proper logging |
| run_hybrid_classification.py | 2025-03-04 | Superseded by llm_tech_classifier.py | llm_tech_classifier.py |
| test_openai_connection.py | 2025-03-04 | Utility script for one-time testing | OpenAI API integration in backend/app/services |
| test_llm_analyzer.py | 2025-03-04 | Test script with no active usage | backend/app/services/llm_analyzer.py |
| test_keyword_classifier.py | 2025-03-04 | Test script with no references in codebase | backend/tests/integration tests |
| check_no_matches.py | 2025-03-04 | One-off utility script with no references | llm_tech_classifier.py with --retry-no-match flag |
| run_batch_classification.ps1 | 2025-03-04 | PowerShell script not compatible with Linux environment | run_classifier.sh |
| partner_analysis.py.bak | 2025-03-04 | Backup file | partner_analysis.py |
| ai_tech_classifier.py.duplicate | 2025-03-04 | Duplicate of existing deprecated file | llm_tech_classifier.py |
| test_llm_classification.py | 2025-03-04 | Superseded by newer implementation | llm_tech_classifier.py |
| create_partner_analysis_doc.py | 2025-03-04 | Limited functionality | analyze_partner_results.py |
| test_batch.py | 2025-03-04 | Testing script incorporated into main functionality | llm_tech_classifier.py |

## Previously Deprecated Scripts

| Script | Deprecated Date | Reason | Replacement |
|--------|----------------|--------|-------------|
| enhanced_vendor_analysis.py | 2025-02-28 | Replaced by partner_analysis.py | partner_analysis.py |
| ai_tech_classifier.py | Unknown | Replaced by llm_tech_classifier.py | llm_tech_classifier.py |

## Note

If you need to reference these scripts, please be aware that they may contain outdated code, bugs, or incompatibilities with the current system. Use them only for reference purposes. 