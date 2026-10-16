# Bioinformatics AI Research Assistant

A Streamlit prototype for exploring scientific literature on artificial intelligence, machine learning, bioinformatics, and drug discovery.

The application retrieves relevant PubMed papers and uses semantic search, cross-encoder reranking, hybrid ranking, evidence extraction, quotation verification, and a large language model to generate answers.

## Features

- Semantic retrieval using sentence embeddings
- Cross-encoder reranking and Reciprocal Rank Fusion
- Evidence extraction from paper abstracts
- Verification of extracted quotations against source text
- AI-generated answers with links to scientific sources
- Retrieval rankings and relevance scores

## Data

The prototype uses a local CSV file containing 41 scientific papers. Its answers depend on the papers available in that dataset and may omit relevant literature.

## Limitations

This is an experimental research and portfolio prototype. Generated answers can contain errors or unsupported interpretations. Verify scientific claims against the original publications.

## Run locally

1. Install the dependencies listed in `requirements.txt`.
2. Set the `GROQ_API_KEY` environment variable.
3. Run `python -m streamlit run app.py`.