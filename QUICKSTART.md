# Dell-AITC Quick Start Guide

## Overview
Dell-AITC is an AI-driven technology categorization system that evaluates and classifies federal AI use cases. This guide will help you get started with development and maintenance.

## Prerequisites
- Python 3.9 or higher
- Docker Desktop
- Neo4j (running in Docker)
- Git

## Initial Setup

1. **Clone the Repository**
```bash
git clone <repository-url>
cd Dell-AITC
```

2. **Install Dependencies**
```bash
pip install -r scripts/requirements.txt
```

3. **Verify Neo4j Setup**
- Ensure Docker is running
- Neo4j should be accessible at:
  - Browser interface: http://localhost:7474
  - Bolt connection: bolt://localhost:7687
  - Default credentials: neo4j/kuxFc8HN

## Project Management

We provide a management script to handle common tasks. Run it from the project root:

```bash
python scripts/manage.py help
```

### Common Commands

1. **Database Management**
```bash
# Reset Neo4j database (creates backup first)
python scripts/manage.py reset-db
```

2. **Code Quality**
```bash
# Format code
python scripts/manage.py format

# Check code quality (runs black, flake8, mypy)
python scripts/manage.py check
```

3. **Testing**
```bash
# Run unit tests
python scripts/manage.py test

# Run all tests including integration
python scripts/manage.py test --integration

# Run tests with coverage report
python scripts/manage.py test --coverage
```

## Development Workflow

### 1. Starting a New Feature
```bash
# Create and switch to new branch
git checkout -b feature/your-feature-name

# Reset database if needed
python scripts/manage.py reset-db
```

### 2. During Development
```bash
# Format code regularly
python scripts/manage.py format

# Run unit tests to verify changes
python scripts/manage.py test
```

### 3. Before Committing
```bash
# Format code
python scripts/manage.py format

# Check code quality
python scripts/manage.py check

# Run unit tests
python scripts/manage.py test
```

### 4. Before Creating Pull Request
```bash
# Run full test suite
python scripts/manage.py test --integration

# Check test coverage
python scripts/manage.py test --coverage
```

## Project Structure

```
Dell-AITC/
├── backend/              # FastAPI backend
│   ├── app/             # Main FastAPI app
│   │   ├── api/         # API endpoints
│   │   ├── models/      # Pydantic models
│   │   ├── services/    # Business logic
│   │   └── config.py    # Configuration
├── scripts/             # Management scripts
│   ├── database/        # Database management
│   │   ├── init/       # Initialization scripts
│   │   ├── loaders/    # Data loading scripts
│   │   └── utils/      # Database utilities
│   ├── analysis/        # Analysis scripts
│   ├── tests/          # Test files
│   └── manage.py       # Management script
├── data/                # Data storage
│   ├── input/          # Raw input files
│   └── output/         # Processed results
└── docs/               # Documentation
```

## Neo4j Database

### Connection Details
- **Host**: localhost
- **Browser Port**: 7474
- **Bolt Port**: 7687
- **Default Username**: neo4j
- **Default Password**: kuxFc8HN

### Database Management
1. **Backup Location**: 
   - Backups are stored in `D:\Docker\neo4j_backup_<timestamp>`
   - Created automatically before reset

2. **Reset Process**:
   - Stops Neo4j container
   - Creates backup
   - Clears data
   - Restarts container
   - Verifies connection

## Troubleshooting

### Common Issues

1. **Neo4j Connection Failed**
   - Verify Docker is running
   - Check if Neo4j container is up
   - Verify ports 7474 and 7687 are not in use
   - Check logs: `docker logs neo4j`

2. **Test Failures**
   - Run with `-v` flag for verbose output:
     ```bash
     python scripts/manage.py test -v
     ```
   - Check test logs in `scripts/tests/logs`

3. **Code Quality Issues**
   - Run format first: `python scripts/manage.py format`
   - Check specific issues: `python scripts/manage.py check`

### Getting Help
1. Check the documentation in `docs/`
2. Run `python scripts/manage.py help`
3. Review relevant test files in `scripts/tests/`

## Next Steps
1. Review the full documentation in `docs/`
2. Explore the test suite in `scripts/tests/`
3. Check out the schema documentation in `docs/neo4j_schema/`

## Best Practices
1. Always create backups before major changes
2. Run tests frequently during development
3. Keep code formatted and linted
4. Update documentation when adding features
5. Add tests for new functionality 