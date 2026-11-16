# Badil

Badil is a Tunisian medicine safety and retrieval project built around an MCP server. It helps users identify safer medicine alternatives, look up product details, compare prices, and query official leaflet information while keeping safety constraints and citations in the response.

## What the project does

The project focuses on three main use cases:

- Medicine lookup: product card, active ingredient, indication, price, availability, and CNAM prior-authorization status.
- Safe substitution: find safer alternatives with risk tiers and pharmacist/doctor validation requirements.
- Official leaflet retrieval: answer questions using official BDPM leaflet sections and cited passages.

Key capabilities exposed by the MCP server include:

- `find_substitute(...)`
- `lookup_medicine(...)`
- `cheaper_alternatives(...)`
- `medicines_for_symptom(...)`
- `leaflet_section(...)`
- `ask_leaflet(...)`

## Project structure

- `badil/` — application code
  - `crawl/` — crawlers for medicine data sources
  - `embedding/` — vector embedding and index creation
  - `graph/` — Neo4j access and graph queries
  - `ingestion/` — ingestion and data import logic
  - `linking/` — matching and linking rules
  - `retrieval/` — name resolution, indexing, and search
  - `safety/` — patient safety checks and rules
  - `scorer/` — equivalence scoring and model logic
  - `sync/` — scheduled sync tasks
  - `tribunal/` — safety tribunal, citation generation, and reasoning
- `data/` — prepared data, models, raw sources, and indexes
- `tests/` — focused regression and safety checks
- `requirements.txt` — Python dependencies

## Requirements

- Python 3.10+
- Access to a Neo4j instance
- GROQ API key for the LLM-based translation and tribunal logic
- Local or network access to the project data directories

## Setup

1. Create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .\.venv\Scripts\activate  # Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables. The app expects a `.env` file with values such as:

   ```env
   NEO4J_URI=your_neo4j_uri
   NEO4J_USERNAME=your_neo4j_username
   NEO4J_PASSWORD=your_neo4j_password
   GROQ_API_KEY=your_groq_api_key
   ```

   The project uses `python-dotenv` to load these values at runtime.

## Running the MCP server

From the project root:

```bash
python badil\mcp_server.py
```

or:

```bash
python -m badil.mcp_server
```

This starts the MCP server using stdio, so it can be used by an MCP client such as Claude Desktop, VS Code, or another compatible tool.

## Data and indexing

The repository includes prepared data under `data/` and relies on generated indexes for embeddings and graph lookups. If you are initializing the project from scratch, make sure the Neo4j database, Qdrant storage, and model/data artifacts exist before invoking the retrieval and tribunal tools.

## Testing

Run the project tests with:

```bash
pytest
```

## Notes

- The project is not a substitute for professional medical advice.
- Safety checks and substitute recommendations are designed to be conservative and to require pharmacist or doctor validation for higher-risk cases.
- The service includes citations and explicit validation requirements in its structured output.
