# Deprecated Content - Dell-AITC

This directory contains historical versions of the Dell-AITC system components that have been deprecated. These files are maintained for reference but should not be used for current development.

## Version History

### v1 (Deprecated: February 2024)
Original implementation with the following key characteristics:
- Complex keyword structure with 5 different types
- Direct zone relationships without proper constraints
- Pre-2024 AI technology categories
- Original schema design without clear domain separation

#### Key Files
- `scripts/fed_use_case_ai_classifier.py`: Original classifier implementation
- `scripts/fed_use_case_ai_importer.py`: Original data import logic
- `docs/neo4j_schema_documentation.md`: Original schema documentation
- `scripts/generate_neo4j_schema_docs.py`: Original documentation generator

### Migration Notes

If you have scripts referencing the old schema:
1. Check the current schema in `/docs/neo4j/schema_planning/SCHEMA_MASTER.md`
2. Use new node and relationship structures defined there
3. Update any custom scripts to use the simplified keyword structure
4. Refer to the new classifier implementation for AI technology categorization

### Reference Only
These files are kept for:
- Historical reference
- Understanding previous implementation decisions
- Migration assistance
- Audit purposes

DO NOT use these files for new development. Always refer to current documentation in the main project directories. 