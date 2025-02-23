# Dell-AITC Cursor Setup Guide

## Overview
This guide helps you set up and work with the Dell-AITC project in Cursor IDE, following our established conventions and rules.

## Initial Setup in Cursor

1. Open the project in Cursor:
   ```
   File -> Open Folder -> Select Dell-AITC directory
   ```

2. Allow Cursor to index the project (this may take a few minutes)

3. Verify Cursor can access key files:
   ```powershell
   # Check essential files
   Get-Content .cursorrules
   Get-Content .env.example
   Get-Content README.md
   ```

## Environment Setup

1. Create and activate conda environment:
   ```powershell
   # Create environment with Python 3.10
   conda create -n dell-aitc python=3.10
   conda activate dell-aitc
   ```

2. Install dependencies:
   ```powershell
   # Backend dependencies
   Set-Location backend
   pip install -r requirements.txt
   
   # Frontend dependencies (if needed)
   Set-Location ../frontend
   npm install
   ```

3. Configure environment:
   ```powershell
   # Copy environment template
   Set-Location ..
   Copy-Item .env.example .env
   
   # Edit .env with your settings
   # Required variables:
   # - NEO4J_URI
   # - NEO4J_USER
   # - NEO4J_PASSWORD
   # - OLLAMA_URL (for local AI)
   # - OPENAI_API_KEY (optional fallback)
   ```

## Database Setup

1. Verify Neo4j service:
   ```powershell
   # Check Neo4j status
   Get-Service neo4j
   
   # Start if needed
   Start-Service neo4j
   ```

2. Initialize database:
   ```powershell
   # Navigate to management scripts
   Set-Location backend/app/services/database/management
   
   # Run initialization (creates schema and loads data)
   python recover_from_nuke.py
   ```

3. Verify initialization:
   ```powershell
   # Run verification script
   python scripts/verify_initialization.py
   
   # This checks:
   # - Schema constraints and indexes
   # - Node counts and relationships
   # - Data integrity
   # - Keyword relevance properties
   ```

## Development Guidelines

1. TypeScript/Frontend:
   - Use interfaces over types
   - PascalCase for components (e.g., `FormWizard.tsx`)
   - Lowercase with dashes for directories
   - Use Shadcn UI components
   - Follow React Context patterns
   - Implement proper cleanup in hooks

2. Python/Backend:
   - Follow documentation block standards
   - Use async/await patterns
   - Implement proper error handling
   - Follow FastAPI best practices
   - Use Neo4j service for database operations

3. Schema Changes:
   - Follow schema change process in `.cursorrules`
   - Document changes in appropriate files
   - Use schema version manager for updates
   - Test changes thoroughly
   - Update all affected components

4. Git Workflow:
   - Use specified commit message prefixes
   - Keep messages under 72 characters
   - Reference issues when applicable
   - Follow branch naming conventions

## Cursor-Specific Tips

1. Code Navigation:
   - `Ctrl+P`: Quick file open
   - `Ctrl+Shift+F`: Global search
   - `F12`: Go to definition
   - `Alt+F12`: Peek definition

2. AI Assistant Usage:
   - Reference `.cursorrules` for project context
   - Ask for help with schema changes
   - Get guidance on code conventions
   - Request documentation help

3. Terminal Commands:
   - Always use PowerShell syntax
   - Use proper cmdlets (e.g., `Set-Location` vs `cd`)
   - Follow Windows path conventions
   - Use proper error handling

4. Common Tasks:
   ```powershell
   # Clear classifications
   python scripts/clear_classifications.py
   
   # Run classifications
   python scripts/test_llm_classification.py --all `
     --batch-size 25 `
     --checkpoint-file data/checkpoints/classification_progress.json `
     --model "your-model-name" `
     --ollama-url "your-ollama-url"
   
   # Verify database state
   python scripts/verify_initialization.py
   ```

## Troubleshooting

1. Database Issues:
   - Check Neo4j service status
   - Verify connection settings in `.env`
   - Check logs in `logs/database/`
   - Run verification script

2. Classification Issues:
   - Verify Ollama is running
   - Check model availability
   - Review classification logs
   - Verify data file presence

3. Environment Issues:
   - Confirm conda environment activation
   - Verify all dependencies installed
   - Check Python path settings
   - Validate environment variables

4. Common Solutions:
   ```powershell
   # Restart Neo4j
   Restart-Service neo4j
   
   # Clear conda environment
   conda deactivate
   conda remove -n dell-aitc --all
   conda create -n dell-aitc python=3.10
   
   # Reset database
   python scripts/clear_classifications.py
   python backend/app/services/database/management/recover_from_nuke.py
   ```

## Next Steps

1. Review Documentation:
   - `docs/neo4j/` for database schema
   - `docs/fed_use_case/` for use case structure
   - `QUICKSTART.md` for common tasks
   - `CHEATSHEET.md` for useful commands

2. Start Development:
   - Follow coding standards
   - Use proper Git workflow
   - Document all changes
   - Test thoroughly

3. Get Help:
   - Use Cursor's AI assistant
   - Reference `.cursorrules`
   - Check documentation
   - Follow troubleshooting guides 