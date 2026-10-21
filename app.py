import os

import pandas as pd
import streamlit as st
from groq import Groq, RateLimitError
from sentence_transformers import SentenceTransformer, CrossEncoder

from functions import (
    search_pubmed,
    prepare_papers,
    prepare_embeddings,
    rank_by_embedding,
    rank_by_cross_encoder,
    hybrid_retrieve,
    build_retrieval_context,
    extract_evidence,
    verify_extracted_quotations,
    build_evidence_table,
    generate_final_answer,
    build_references,
    replace_pmid_citations,
    get_author_year_citation,
)


# Configurações gerais do aplicativo
st.set_page_config(
    page_title="Scientific Literature Research Assistant",
    page_icon="🧬",
    layout="wide"
)

MODEL_NAME = "openai/gpt-oss-120b"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K = 5
RRF_K = 60
PAPERS_PATH = "pubmed_papers.csv"

MAX_AI_QUERIES = 5
if "ai_queries" not in st.session_state:
    st.session_state.ai_queries = 0


#-----------------------------------------------------------------------------------------


@st.cache_resource
def load_embedding_model():
    # Carrega o modelo usado para representar os textos numericamente.
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_reranker():
    # Carrega o modelo que reavalia a relevância dos artigos.
    return CrossEncoder(RERANKER_MODEL_NAME)


@st.cache_data
def load_papers():
    # Carrega os artigos salvos no arquivo CSV.
    papers = pd.read_csv(PAPERS_PATH)

    # Cria o texto usado na busca caso a coluna ainda não exista.
    if "text" not in papers.columns:
        papers["text"] = (
            papers["title"].fillna("").astype(str)
            + ". "
            + papers["abstract"].fillna("").astype(str)
        )

    # Valida os dados antes de utilizá-los.
    return prepare_papers(papers)


@st.cache_resource
def load_paper_embeddings(_papers, _embedding_model):
    # Gera os embeddings uma única vez por execução do aplicativo.
    return prepare_embeddings(
        papers=_papers,
        embedding_model=_embedding_model
    )


#-----------------------------------------------------------------------------------------



# Apresentação do aplicativo
st.title("🧬 Scientific Literature Research Assistant")
st.write(
    "Search PubMed for biomedical and life-science literature. "
)

# Obtém a chave da API sem exibi-la na tela.
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error(
        "Groq API key not found. Set the GROQ_API_KEY environment variable "
        "before running the application."
    )
    st.stop()

# Inicializa o cliente da API.
client = Groq(api_key=api_key)

# Carrega os artigos da coleção curada.
try:
    papers_curated = load_papers()
except Exception as e:
    st.error("Não foi possível carregar a coleção curada de artigos.")
    st.exception(e)
    st.stop()

# Cria as duas áreas de pesquisa.
tab_curated, tab_pubmed = st.tabs(
    ["📚 ML & Bioinformatics Preselected Papers", "🔎 Search PubMed (for broader biomedical and life-science searches)"]
)


#-----------------------------------------------------------------------------------------
# Função compartilhada pelas duas áreas de pesquisa
#-----------------------------------------------------------------------------------------


def run_research(question, papers, usar_cache=False):
    """Executa a recuperação de artigos, a extração de evidências e a resposta."""

    if papers.empty:
        st.warning("Nenhum artigo disponível para responder à pergunta.")
        return

    with st.spinner("Loading AI models..."):
        embedding_model = load_embedding_model()
        reranker = load_reranker()

        # Reutiliza os embeddings em cache para a coleção curada.
        if usar_cache:
            paper_embeddings = load_paper_embeddings(
                papers, embedding_model
            )
        else:
            # Os embeddings da busca no PubMed são gerados temporariamente.
            paper_embeddings = prepare_embeddings(
                papers=papers,
                embedding_model=embedding_model
            )

    with st.spinner("Finding relevant papers..."):
        # Classifica os artigos pela similaridade semântica.
        embedding_results = rank_by_embedding(
            query=question,
            papers=papers,
            paper_embeddings=paper_embeddings,
            embedding_model=embedding_model
        )

        # Reavalia os artigos com o cross-encoder.
        cross_encoder_results = rank_by_cross_encoder(
            query=question,
            papers=papers,
            cross_encoder=reranker
        )

        # Combina os dois rankings usando Reciprocal Rank Fusion.
        hybrid_results = hybrid_retrieve(
            embedding_results=embedding_results,
            cross_encoder_results=cross_encoder_results,
            top_k=min(TOP_K, len(papers)),
            rrf_k=RRF_K
        )

        retrieval_context = build_retrieval_context(hybrid_results)

    with st.spinner("Extracting evidence from the papers..."):
        evidence = extract_evidence(
            question=question,
            context=retrieval_context,
            client=client,
            model_name=MODEL_NAME
        )

        checked_quotes = verify_extracted_quotations(
            evidence=evidence,
            papers=papers
        )

    evidence_table = build_evidence_table(
        checked_quotes=checked_quotes,
        papers=papers
    )

#-----------------------------------------------------------------------------------------

    st.subheader("Final answer")

    st.warning(
        "The text below is an AI-generated response based on retrieved scientific "
        "papers. It may contain inaccuracies or unsupported interpretations. "
        "Check the original publications before relying on its claims."
    )

    with st.spinner("Generating the final answer..."):
        final_answer = generate_final_answer(
            query=question,
            retrieved_papers=hybrid_results,
            evidence=evidence,
            checked_quotes=checked_quotes,
            client=client,
            model_name=MODEL_NAME
        )


    answer_display = final_answer.replace("【PMID:", "[PMID:")
    answer_display = answer_display.replace("】", "]")
    answer_display = replace_pmid_citations(
        answer_display,
        hybrid_results
    )

    st.markdown(answer_display)

#-----------------------------------------------------------------------------------------

    # Exibe cada evidência junto com sua verificação.
    papers_by_pmid = {
        str(row.get("pmid", "")).strip(): row
        for _, row in hybrid_results.iterrows()
    }

    with st.expander("Evidence and quotation verification"):
        
        if evidence_table.empty:
            st.info(
                "No relevant evidence was extracted from the retrieved papers. "
                "This does not necessarily mean that no research exists on the topic. "
                "The relevant studies may not have been retrieved, or their abstracts "
                "may not contain sufficient information to answer the question. "
                "Try refining your search or using more specific keywords."
            )
        else:

            for _, row in evidence_table.iterrows():
                paper = {
                    "authors": row.get("authors", []),
                    "year": row.get("year", "")
                }
                citation = get_author_year_citation(paper)

                title = row["Paper"]
                heading = f"{title} ({citation})" if citation else title

                with st.container(border=True):
                    st.markdown(f"**Paper:** {heading}")

                    st.markdown("**Quotation selected by the LLM**")
                    st.markdown(f"> {row['Quotation']}")

                    st.markdown("**Abstract used for verification**")
                    st.markdown(f'> {row["Abstract passage"]}')

                    st.markdown(
                        f"**Verification:** {row['Verification']}  \n"
                        f"**Source matched:** {row['Source matched']}"
                    )
                    st.markdown(f"[Open source paper]({row['url']})")

#-----------------------------------------------------------------------------------------
    # # # Funcionalidade desativada, pois essas informações 
    # # # já são mostradas no expander anterior

    # # Mostra as referências
    # with st.expander("📚 References"):
    #     references = build_references(hybrid_results)

    #     for _, reference in references.iterrows():
    #         st.markdown(
    #             f"**{reference['Citation']}** — "
    #             f"[{reference['Title']}]({reference['URL']})"
    #         )

    #         authors = reference["Authors"]
    #         if isinstance(authors, list) and authors:
    #             st.caption("Authors: " + ", ".join(authors))

    #         if reference["Year"]:
    #             st.caption(f"Publication year: {reference['Year']}")

    #         st.caption(f"PMID: {reference['PMID']}")

#-----------------------------------------------------------------------------------------

    # Exibe os artigos recuperados e suas pontuações de relevância.
    with st.expander("Retrieved papers and relevance scores"):
        colunas = [
            "hybrid_rank",
            "title",
            "embedding_score",
            "embedding_rank",
            "cross_score",
            "cross_rank",
            "rrf_score"
        ]

        df_to_show = hybrid_results[colunas].round(4).rename(columns={
            "hybrid_rank": "Final rank",
            "title": "Paper",
            "embedding_score": "Embedding score",
            "embedding_rank": "Embedding rank",
            "cross_score": "Cross-encoder score",
            "cross_rank": "Cross-encoder rank",
            "rrf_score": "RRF score"
        })

        # Formata as pontuações com quatro casas decimais.
        float_cols = [
            "Embedding score",
            "Cross-encoder score",
            "RRF score"
        ]
        df_to_show[float_cols] = df_to_show[float_cols].map(
            lambda x: f"{x:.4f}"
        )

        st.table(df_to_show.set_index("Final rank"))

        st.markdown("**How to interpret these results**")
        st.markdown(
            """
            - **Final rank:** the paper's position after combining both ranking methods.
            - **Paper:** the title of the article.
            - **Embedding score/rank:** measures how closely the paper's meaning matches your question. Higher scores indicate greater semantic similarity.
            - **Cross-encoder score/rank:** a second model evaluates the question and paper together to estimate their relevance. Higher scores generally indicate greater relevance.
            - **RRF score (Reciprocal Rank Fusion):** combines the positions assigned by both methods. Papers ranked highly by both tend to receive higher RRF scores.
            """
        )


#-----------------------------------------------------------------------------------------
# Aba 1: coleção curada de artigos
#-----------------------------------------------------------------------------------------


with tab_curated:
    st.subheader("Curated Collection")
    st.write(
        f"Search your research question across the {len(papers_curated)} "
        "articles in the curated collection."
    )

    with st.form("form_curated"):
        question_curated = st.text_input(
            "Your research question",
            placeholder="How is machine learning used to identify drug candidates?"
        )
        submit_curated = st.form_submit_button(
            "Search curated papers and generate answer",
            type="primary"
        )

    if submit_curated:
        if not question_curated.strip():
            st.warning("Enter a research question.")
        elif st.session_state.ai_queries >= MAX_AI_QUERIES:
            st.warning("Session limit reached. Please try again later.")
        else:
            st.session_state.ai_queries += 1
            try:
                st.success(
                    f"Using {len(papers_curated)} papers from the curated collection."
                )

                # Mostra os artigos encontrados antes de gerar a resposta.
                with st.expander("Articles in the curated collection", expanded=False):
                    for _, paper in papers_curated.iterrows():
                        citation = get_author_year_citation(paper)
                        label = f"[{paper['title']}]({paper['url']})"
                        if citation:
                            label += f" ({citation})"
                        st.markdown(f"- {label}")
                        
                run_research(
                    question=question_curated.strip(),
                    papers=papers_curated,
                    usar_cache=True
                )

            except RateLimitError:
                st.error("The AI service has reached its usage limit. Please try again later.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.exception(e)


#-----------------------------------------------------------------------------------------
# Aba 2: busca de novos artigos no PubMed
#-----------------------------------------------------------------------------------------


with tab_pubmed:
    st.subheader("Search PubMed")
    st.write(
        "Search PubMed using your own query. The app retrieves up to 50 "
        "articles with abstracts. Search results are temporary and do not "
        "modify the preselected papers."
    )

    with st.form("form_pubmed"):
        pubmed_query = st.text_input(
            "PubMed search query",
            placeholder='("machine learning" AND bioinformatics) OR "drug discovery"'
        )

        question_pubmed = st.text_input(
            "Your research question",
            placeholder="What methods and findings are reported in these studies?"
        )

        submit_pubmed = st.form_submit_button(
            "Search PubMed and generate answer",
            type="primary"
        )

    if submit_pubmed:
        if not pubmed_query.strip():
            st.warning("Enter a PubMed search query.")
        elif not question_pubmed.strip():
            st.warning("Enter a separate research question.")
        elif st.session_state.ai_queries >= MAX_AI_QUERIES:
            st.warning("Session limit reached. Please try again later.")
        else:
            try:
                with st.spinner("Searching PubMed for new articles..."):
                    # Recupera artigos sem alterar a coleção salva.
                    existing_pmids = papers_curated["pmid"].astype(str).tolist()
                    papers_pubmed = search_pubmed(
                        query=pubmed_query.strip(),
                        existing_pmids=existing_pmids,
                        retmax=50
                    )

                if papers_pubmed.empty:
                    st.warning(
                        "No new articles with abstracts were found. Try another "
                        "query or broaden your search."
                    )
                else:
                    st.session_state.ai_queries += 1
                    st.success(
                        f"Retrieved {len(papers_pubmed)} new articles with abstracts."
                    )

                    # Mostra os artigos encontrados antes de gerar a resposta.
                    with st.expander("Articles retrieved from PubMed", expanded=False):
                        for _, paper in papers_pubmed.iterrows():
                            citation = get_author_year_citation(paper)
                            title = paper["title"]
                            url = paper["url"]

                            label = f"[{title}]({url})"
                            if citation:
                                label += f" ({citation})"

                            st.markdown(f"- {label}")

                    run_research(
                        question=question_pubmed.strip(),
                        papers=papers_pubmed,
                        usar_cache=False
                    )

            except RateLimitError:
                st.error("The AI service has reached its usage limit. Please try again later.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.exception(e)