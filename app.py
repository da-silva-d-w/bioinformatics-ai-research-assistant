import os

import pandas as pd
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder

from functions import (
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
)


# Configurações gerais do aplicativo
st.set_page_config(
    page_title="Bioinformatics AI Research Assistant",
    page_icon="🧬",
    layout="wide"
)

MODEL_NAME = "openai/gpt-oss-120b"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K = 5
RRF_K = 60
PAPERS_PATH = "pubmed_papers.csv"


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
st.title("🧬 Bioinformatics AI Research Assistant")
st.write(
    "Ask questions about artificial intelligence, machine learning, "
    "bioinformatics, and drug discovery using scientific papers from PubMed."
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

# Campo para a pergunta do usuário.
question = st.text_input(
    "Your research question",
    placeholder="How is machine learning used to identify drug candidates?"
)

# Botão para iniciar a busca e gerar a resposta.
run_search = st.button(
    "Search papers and generate answer",
    type="primary",
    disabled=not question.strip()
)


#-----------------------------------------------------------------------------------------


if run_search:
    try:
        with st.spinner("Loading papers and AI models..."):
            papers = load_papers()
            embedding_model = load_embedding_model()
            reranker = load_reranker()
            paper_embeddings = load_paper_embeddings(
                papers, embedding_model
            )

        st.success(f"Loaded {len(papers)} scientific papers.")

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

        #-----------------------------------------------------------------------------------------

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
            "The text below is an AI-generated response based on retrieved scientific papers. "
            "It may contain inaccuracies or unsupported interpretations. "
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

        st.markdown(final_answer)

        #-----------------------------------------------------------------------------------------

        with st.expander("Evidence and quotation verification"):
                    st.markdown("**Extracted evidence**")
                    st.markdown(evidence)
                    st.markdown("---")
                    st.markdown("**Quotation verification results**")
                    for _, row in evidence_table.iterrows():
                        st.markdown(f"**Quotation:** {row['Quotation']}")
                        st.markdown(f"**Paper:** {row['Paper']}")
                        st.markdown(
                            f"**Verification:** {row['Verification']}  \n"
                            f"**Source matched:** {row['Source matched']}"
                        )
                        st.markdown(f"[Open source paper]({row['url']})")
                        st.divider()

        #-----------------------------------------------------------------------------------------

        with st.expander("Retrieved papers and relevance scores"):

                    # 1. Filtre as colunas e arredonde os valores
                    df_to_show = hybrid_results[
                        [
                            "hybrid_rank",
                            "title",
                            "embedding_score",
                            "embedding_rank",
                            "cross_score",
                            "cross_rank",
                            "rrf_score"
                        ]
                    ].round(4)

                    # 2. Renomeie as colunas (substituindo o "column_config")
                    df_to_show = df_to_show.rename(columns={
                        "hybrid_rank": "Final rank",
                        "title": "Paper",
                        "embedding_score": "Embedding score",
                        "embedding_rank": "Embedding rank",
                        "cross_score": "Cross-encoder score",
                        "cross_rank": "Cross-encoder rank",
                        "rrf_score": "RRF score"
                    })

                    # 3. Formate os números flutuantes para exibir 4 casas decimais (igual ao format="%.4f")
                    # Isso garante que zeros à direita não sumam na tabela estática
                    float_cols = ["Embedding score", "Cross-encoder score", "RRF score"]
                    df_to_show[float_cols] = df_to_show[float_cols].map(lambda x: f"{x:.4f}")

                    # 4. Use st.table. Para esconder o índice, envie apenas os valores (.values) 
                    # e defina as colunas manualmente
                    st.table(df_to_show.set_index("Final rank")) 
                    # Dica: Usar o "Final rank" como índice economiza espaço e remove a coluna padrão do Pandas!


                    st.markdown("**Final rank:** combined position.")                        
                    st.markdown("**Paper:** article title.")
                    st.markdown("**Embedding score/rank:** semantic similarity and its position.")
                    st.markdown("**Cross-encoder score/rank:** relevance score and its position.")
                    st.markdown("**RRF score:** combined ranking score; higher values indicate a higher rank.")

        #-----------------------------------------------------------------------------------------

    # except Exception as error:
    #     st.error(f"An error occurred during retrieval: {error}")

    except Exception as e:
        st.error("An error occurred during retrieval:")
        st.exception(e)


#-----------------------------------------------------------------------------------------

