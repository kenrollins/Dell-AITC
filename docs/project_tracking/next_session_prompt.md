# Next Session Prompt

## Context
We are working on the Dell-AITC system, an AI-driven technology categorization system that evaluates and classifies federal AI use cases against 14 AI technology categories. We have successfully implemented and tested the classification system with enhanced LLM analysis capabilities and are now ready to move towards processing the full database of 2000+ use cases.

## Project Rules
1. Schema Changes: All modifications must follow the schema change process documented in SCHEMA_CHANGES.md
2. Commands: All terminal commands must use PowerShell syntax and conventions for Windows
3. Documentation: Maintain comprehensive documentation of all changes and processes
4. Error Handling: Implement robust error handling with proper logging and recovery mechanisms

## Current State
- Enhanced LLM analyzer with retry logic and increased timeout
- Successful test runs with sample use cases
- Schema v2.2 compatibility verified
- Ready for full database implementation

## Next Session Goals
1. Implement batch processing for large-scale classification
2. Establish monitoring and tracking systems
3. Begin live database integration
4. Set up performance metrics collection

## Required Information
Please provide the following information at the start of the next session:

1. Neo4j Database Configuration:
   ```powershell
   $NEO4J_URI="bolt://localhost:7687"
   $NEO4J_USER="neo4j"
   $NEO4J_PASSWORD="your_password"
   ```

2. OpenAI API Configuration:
   ```powershell
   $OPENAI_API_KEY="your_api_key"
   $OPENAI_ORG_ID="your_org_id"  # if applicable
   ```

3. System Resource Allocation:
   - Dedicated memory allocation
   - CPU core allocation
   - Storage requirements
   - Network bandwidth allocation

4. Processing Requirements:
   - Desired batch size
   - Processing time constraints
   - Error tolerance levels
   - Success criteria

## Starting Commands
To begin the next session, we'll need to:

1. Verify environment:
   ```powershell
   python -c "from backend.app.config import settings; print(settings.dict())"
   ```

2. Check database connection:
   ```powershell
   python -c "from backend.app.dependencies import get_db; db = get_db(); print(db.test_connection())"
   ```

3. Validate OpenAI configuration:
   ```powershell
   python -c "from backend.app.services.llm_analyzer import LLMAnalyzer; analyzer = LLMAnalyzer(); print(analyzer.test_connection())"
   ```

## Initial Questions for Next Session
1. What batch size would you like to start with for processing use cases?
2. Should we implement a priority system for certain use cases?
3. What level of logging detail do you require for the batch processing?
4. How should we handle partial failures during batch processing?
5. What metrics should we track for performance monitoring?

## Suggested Approach
1. Start with a small batch (10-20 use cases) to validate the process
2. Implement progress tracking and monitoring
3. Scale up batch size based on performance metrics
4. Establish error recovery procedures
5. Document results and adjust process as needed

## Success Metrics
1. Classification accuracy (confidence levels)
2. Processing speed (cases per hour)
3. Resource utilization
4. Error rates and recovery success
5. Data consistency and integrity

## Backup Plan
1. Regular database backups
2. Fallback processing options
3. Error recovery procedures
4. Manual review process for failed cases

Please review this information and have the required configurations ready for the next session. We'll begin by implementing the batch processing system and establishing monitoring procedures before starting the full database processing. 