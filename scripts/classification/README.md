# Classification Scripts

This directory contains utility scripts for classifying and analyzing federal AI use cases.

## Scripts

### `ai_tech_classifier.py`
A standalone utility script that evaluates federal AI use cases and classifies them against AI technology categories using a multi-method approach:
- Keyword matching
- Semantic analysis
- LLM verification (Ollama local or OpenAI)

The script works directly with Neo4j for both data access and storage, creating proper relationships between use cases and technology categories.

### Usage

```bash
# Process all unclassified use cases using Ollama (default)
python ai_tech_classifier.py -a

# Process specific number of use cases
python ai_tech_classifier.py -n 50

# Use OpenAI for production run
python ai_tech_classifier.py -a --llm-provider openai

# Dry run with custom batch size
python ai_tech_classifier.py -n 100 --dry-run --batch-size 20
```

### Command Line Arguments

```
-n, --number NUM       Number of use cases to process
-a, --all             Process all unclassified use cases
--dry-run             Run without making database changes
--llm-provider        LLM provider to use (ollama or openai, default: ollama)
--batch-size          Number of cases to process in each batch (default: 10)
```

## Environment Setup

Required environment variables:
```
NEO4J_URI               # Neo4j database URI
NEO4J_USER             # Neo4j username
NEO4J_PASSWORD         # Neo4j password
OPENAI_API_KEY         # OpenAI API key (required if using openai provider)
OLLAMA_BASE_URL        # Ollama base URL (required if using ollama provider)
```

## Output

The script will:
1. Create `AIClassification` relationships in Neo4j with:
   - Classification type (PRIMARY, SECONDARY, RELATED)
   - Confidence scores from each method
   - Justification and analysis
   
2. Generate `UnmatchedAnalysis` nodes for cases that don't match any category:
   - Reason categorization (NOVEL_TECH, IMPLEMENTATION_SPECIFIC, etc.)
   - LLM analysis and suggestions
   - Links to potential new categories
   
3. Log detailed metrics about the classification process:
   - Number of cases processed
   - Success/failure rates
   - Method performance statistics
   - Batch processing times 