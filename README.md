# Dell-AITC (AI Technology Categorization System)

A local AI-driven technology categorization system that evaluates and classifies federal AI use cases against a set of 14 AI technology categories.

## Features
- AI-powered classification of federal use cases
- Knowledge graph storage using Neo4j
- Local AI processing with Ollama (OpenAI API fallback)
- FastAPI backend with async processing
- Modern Next.js frontend with TypeScript

## Tech Stack
- Backend: FastAPI, Python, Neo4j
- AI Processing: Ollama, OpenAI API
- Frontend: Next.js, TypeScript, TailwindCSS
- Database: Neo4j Graph Database
- Infrastructure: Docker

## Project Structure
```
Dell-AITC/
│── backend/              # FastAPI backend
│── frontend/             # Next.js frontend
│── data/                 # Data storage
│── notebooks/            # Jupyter notebooks
│── docs/                 # Documentation
│── scripts/              # Utility scripts
```

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js 16+
- Docker and Docker Compose
- Neo4j Database
- Ollama (for local AI processing)
- Anaconda or Miniconda

### Ubuntu 22.04 Setup

1. Install system dependencies:
```bash
# Update package list
sudo apt update

# Install required packages
sudo apt install -y \
    build-essential \
    curl \
    git \
    docker.io \
    docker-compose

# Install Node.js 16+
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Verify installations
node --version
npm --version
docker --version
docker-compose --version
```

2. Install Anaconda:
```bash
# Download Anaconda installer
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh

# Make installer executable
chmod +x Anaconda3-2024.02-1-Linux-x86_64.sh

# Run installer
./Anaconda3-2024.02-1-Linux-x86_64.sh

# Activate Anaconda in current shell
source ~/.bashrc

# Verify installation
conda --version
```

3. Install Ollama:
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
systemctl --user start ollama

# Verify Ollama is running
curl http://localhost:11434/api/version
```

4. Install Neo4j:
```bash
# Import Neo4j GPG key
curl -fsSL https://debian.neo4j.com/neotechnology.gpg.key | sudo gpg --dearmor -o /usr/share/keyrings/neo4j.gpg

# Add Neo4j repository
echo "deb [signed-by=/usr/share/keyrings/neo4j.gpg] https://debian.neo4j.com stable latest" | sudo tee /etc/apt/sources.list.d/neo4j.list

# Install Neo4j
sudo apt update
sudo apt install -y neo4j

# Start Neo4j service
sudo systemctl start neo4j

# Enable Neo4j service to start on boot
sudo systemctl enable neo4j

# Verify Neo4j is running
sudo systemctl status neo4j
```

### Project Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd Dell-AITC
```

2. Create and activate Anaconda environment:
```bash
# Create new environment
conda create -n dell-aitc python=3.10

# Activate environment
conda activate dell-aitc

# Verify Python version
python --version
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration:
nano .env  # or use your preferred editor
```

4. Install project dependencies:
```bash
# Backend dependencies
cd backend
pip install -r requirements.txt

# Frontend dependencies
cd ../frontend
npm install
```

5. Initialize Neo4j:
```bash
# Set Neo4j password (replace 'your-password' with actual password)
sudo neo4j-admin set-initial-password your-password

# Update .env with the password
# NEO4J_PASSWORD=your-password
```

6. Initialize the database:
```bash
# Clear any existing classifications if needed
cd frontend
npm run dev
```

## Database Initialization and Verification

### Initial Setup

1. Verify Neo4j is running:
   ```bash
   sudo systemctl status neo4j
   ```

2. Set Neo4j password (first time only):
   ```bash
   cypher-shell -u neo4j -p neo4j
   ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO 'your-new-password';
   :exit
   ```

3. Update `.env` with Neo4j credentials:
   ```bash
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your-new-password
   ```

### Database Initialization

4. Initialize database schema and load data:
   ```bash
   cd backend/app/services/database/management
   python recover_from_nuke.py
   ```
   
   This script will:
   - Initialize schema constraints and indexes
   - Load AI technology zones
   - Load AI technology categories
   - Load federal AI inventory data
   - Create relationships between nodes
   - Update keyword relevance scores

### Verification

5. Verify database initialization:
   ```bash
   python scripts/verify_initialization.py
   ```
   
   The verification script checks:
   - Schema constraints and indexes
   - Node counts (Zones, Categories, Use Cases, etc.)
   - Relationship types and counts
   - Data integrity (no orphaned nodes)
   - Keyword relevance properties
   
   If any issues are found, detailed error messages will be displayed.

### Classification

6. Clear existing classifications (if needed):
   ```bash
   python scripts/clear_classifications.py
   ```

7. Run classifications:
   ```bash
   python scripts/test_llm_classification.py --all \
     --batch-size 25 \
     --checkpoint-file data/checkpoints/classification_progress.json \
     --model "your-model-name" \
     --ollama-url "your-ollama-url"
   ```

### Troubleshooting

If verification fails:
1. Check Neo4j logs: `sudo journalctl -u neo4j -n 100`
2. Verify data files exist in `data/input/`:
   - `AI-Technology-zones-v*.csv`
   - `AI-Technology-Categories-v*.csv`
   - `[YYYY]_consolidated_ai_inventory_raw_v*.csv`
3. Check database connection: `cypher-shell -u neo4j -p your-password`
4. Review initialization logs in `logs/recover_from_nuke.log`

## Documentation
- [Neo4j Schema](docs/neo4j/neo4j_schema_documentation.md)
- [Implementation Notes](docs/implementation.md)
- [API Documentation](docs/api.md)

## Contributing
Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

## License
[License details to be added]

## Environment Setup

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Update the `.env` file with your credentials:
   - Set `NEO4J_PASSWORD` to your Neo4j database password
   - (Optional) Set `OPENAI_API_KEY` if using OpenAI as a fallback

3. Install dependencies:
   ```bash
   # Backend
   cd backend
   pip install -r requirements.txt
   
   # Frontend
   cd ../frontend
   npm install
   ```

4. Initialize the database:
   ```bash
   python scripts/clear_classifications.py  # If needed
   ```

## Classification Process

The system uses a multi-step process to classify federal AI use cases:
1. Keyword matching
2. Semantic evaluation
3. LLM-based classification

Classifications are stored in Neo4j with:
- Primary matches (confidence > 0.8)
- Supporting matches (confidence > 0.6)
- Related matches (confidence > 0.4)

## Project Structure

See `docs/` directory for detailed documentation on:
- Neo4j schema
- Federal use case structure
- Classification methodology

### Development Commands (Ubuntu)

1. Start all services:
```bash
# Start Neo4j if not running
sudo systemctl start neo4j

# Start Ollama if not running
systemctl --user start ollama

# Start backend (in dell-aitc conda environment)
conda activate dell-aitc
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Start frontend (in another terminal)
cd frontend
npm run dev
```

2. Monitoring logs:
```bash
# Neo4j logs
sudo journalctl -u neo4j -f

# Ollama logs
journalctl --user -u ollama -f

# Application logs are in logs/ directory
```

3. Managing services:
```bash
# Neo4j
sudo systemctl start neo4j    # Start
sudo systemctl stop neo4j     # Stop
sudo systemctl restart neo4j  # Restart

# Ollama
systemctl --user start ollama    # Start
systemctl --user stop ollama     # Stop
systemctl --user restart ollama  # Restart
```

4. Database maintenance:
```bash
# Clear classifications
conda activate dell-aitc
python scripts/clear_classifications.py

# Backup Neo4j database
sudo neo4j-admin dump --database=neo4j --to=/path/to/backup/neo4j-backup.dump

# Restore Neo4j database
sudo neo4j-admin load --from=/path/to/backup/neo4j-backup.dump --database=neo4j --force
```
