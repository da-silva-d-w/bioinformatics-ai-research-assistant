# Scientific Literature Research Assistant

**Live demo:** https://bioinformatics-ai-research-assistant.streamlit.app/

A Streamlit application for exploring scientific literature on artificial intelligence, machine learning, bioinformatics, and drug discovery. It combines semantic search and relevance ranking with LLM-based answer generation to help users discover and investigate potentially relevant scientific papers.

## Features

* **Curated literature search:** Explore a local collection of preselected papers on machine learning and bioinformatics.
* **PubMed search:** Retrieve up to 50 new articles with abstracts using the PubMed API.
* **Semantic retrieval:** Represent scientific questions and papers using sentence embeddings.
* **Cross-encoder reranking:** Refine relevance estimates by evaluating questions and papers together.
* **Hybrid ranking:** Combine retrieval rankings using Reciprocal Rank Fusion (RRF).
* **Evidence extraction:** Identify relevant statements in retrieved abstracts to support generated answers.
* **Quotation verification:** Check extracted quotations against source abstracts and identify matching source papers.
* **AI-generated answers:** Produce literature-based explanations with author–year citations and links to original publications.
* **Retrieval transparency:** Display retrieved papers, ranking results, and relevance scores.

## Technologies

* Python
* Streamlit
* PubMed E-utilities API
* Sentence Transformers
* Cross-encoder models
* Groq API and large language models
* Pandas

## Data and sources

The curated literature collection consists of a local CSV file containing preselected scientific papers. The PubMed search tab retrieves additional articles dynamically; these results are temporary and do not modify the curated collection.

The application uses article abstracts to retrieve evidence and generate answers. Access to full-text publications depends on their availability through the linked sources.

## Limitations

This is an experimental research and portfolio prototype, not a substitute for systematic literature reviews or expert scientific judgment.

* Search results depend on the query, available papers, and retrieval and ranking methods.
* Generated answers may contain errors, omissions, or unsupported interpretations.
* Quotation verification checks extracted text against available abstracts; it does not establish that the scientific interpretation is correct.
* PubMed API availability and LLM provider rate limits may affect searches and answer generation.
* Always consult the original publications before relying on scientific claims.

## Run locally

1. Clone the repository.
2. Install the dependencies listed in `requirements.txt`.
3. Set the `GROQ_API_KEY` environment variable.
4. Run the application:

   ```bash
   python -m streamlit run app.py
   ```
