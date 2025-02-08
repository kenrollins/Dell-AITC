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

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd Dell-AITC
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Start services with Docker:
```bash
docker-compose up -d
```

4. Install backend dependencies:
```bash
cd backend
pip install -r requirements.txt
```

5. Install frontend dependencies:
```bash
cd frontend
npm install
```

## Development

### Running the Backend
```bash
cd backend
uvicorn app.main:app --reload
```

### Running the Frontend
```bash
cd frontend
npm run dev
```

## Documentation
- [Neo4j Schema](docs/neo4j/neo4j_schema_documentation.md)
- [Implementation Notes](docs/implementation.md)
- [API Documentation](docs/api.md)

## Contributing
Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

## License
[License details to be added]
