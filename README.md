# UMLS Knowledge Graph System (CRKGS)

A multi-modal knowledge graph system built to measure the relevance between clinical concepts (UMLS/OMOP) and specific medical domains using hybrid search combining graph topology, semantic embeddings, and expert-curated taxonomical weights.

## Architecture

The system follows a 3-layer architecture:

- **Layer 1**: Raw ground truth (concept nodes, relationship edges, ancestor edges)
- **Layer 2**: Metric sidecar (computed features: centrality, embeddings, authority scores)
- **Layer 3**: Relevance scoring (hybrid search combining graph + vector + metrics)

## Features

- **Data Ingestion**: Process large UMLS CSV files using Polars with chunking support
- **Feature Engineering**: Compute graph metrics (eigenvector centrality, hierarchy depth) and semantic embeddings
- **Hybrid Search**: Combine vector similarity search with graph traversal
- **Relevance Scoring**: Calculate non-linear scores [0, 1] based on structural, semantic, and density metrics
- **Auditability**: Track data transformations and metric versions using Dagster

## Tech Stack

- **Orchestration**: Dagster (Software-Defined Assets)
- **Processing**: Polars (Memory-efficient CSV/Dataframe manipulation)
- **Database**: SurrealDB (Multi-model: Graph + Vector + Document)
- **Embeddings**: Local pipeline (Sentence-Transformers/PyTorch)
- **Language**: Python 3.12+

## Project Structure

```
UMLS_KG/
├── src/
│   ├── ingestion/          # Data loading and cleaning
│   ├── database/           # SurrealDB operations
│   ├── transformers/       # Data transformations
│   ├── features/           # Feature engineering
│   ├── scoring/            # Relevance scoring
│   └── search/             # Hybrid search
├── assets/                 # Dagster assets
├── pipelines/              # Dagster pipeline definitions
├── config/                 # Configuration files
├── tests/                  # Test suite
└── data/                   # UMLS CSV files
```

## Setup

1. Install dependencies using `uv`:
```bash
uv sync
```

2. Start SurrealDB:
```bash
surreal start --log debug --user root --pass root memory
```

3. Run the Dagster pipeline:
```bash
dagster dev
```

## Usage

### Running the Pipeline

The pipeline processes data in the following order:

1. `umls_raw_load` - Load and clean CSV files
2. `surreal_ingest_*` - Ingest raw data into SurrealDB
3. `graph_metrics_calc` - Calculate centrality and depth
4. `concept_embedding_gen` - Generate embeddings
5. `authority_scores_calc` - Calculate vocabulary authority
6. `surreal_upsert_metrics` - Sync metrics to SurrealDB
7. `relevance_scores_calc` - Compute relevance scores
8. `audit_report` - Generate coverage report

### Hybrid Search

```python
from src.search.hybrid_search import search_concepts_by_domain

# Search by domain cluster
results = search_concepts_by_domain("Condition", limit=10, min_score=0.5)
```

## Testing

Run tests with pytest:

```bash
pytest tests/
```

## Configuration

- `config/surreal_schema.surql` - SurrealDB schema definitions
- `config/dagster.yaml` - Dagster configuration
- `config/optimization.surql` - Database optimization queries

## License

See LICENSE file for details.

