from dotenv import dotenv_values

from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.postprocessor.jinaai_rerank import JinaRerank
from llama_index.llms.sambanovasystems import SambaNovaCloud

config = dotenv_values("backend/.env")

# MODELS
EMBEDDING_MODEL: str = "jina-embeddings-v3"

LLM_MODEL: str = "Meta-Llama-3.2-3B-Instruct"

RERANK_MODEL: str = "jina-reranker-v2-base-multilingual"

# text embeddings from pdf
text_embed_model = JinaEmbedding(
    api_key=config["JINA_API_KEY"],
    model=EMBEDDING_MODEL,
    task="retrieval.passage",
    dimension=512,
    late_chunking=True,
    embedding_type="float"
    )

# text embeddings from user input
query_embed_model = JinaEmbedding(
    api_key=config["JINA_API_KEY"],
    model=EMBEDDING_MODEL,
    task="retrieval.query",
    dimension=512,
    late_chunking=False,
    embedding_type="float"
    )

# Meta-Llama-3.2-3B-Instruct
llm = SambaNovaCloud(sambanova_url=config["SAMBANOVA_URL"],
                     sambanova_api_key=config["SAMBANOVA_API_KEY"],
                     model=LLM_MODEL,
                     max_tokens=1024,
                     temperature=0.2
                     )

# jina-reranker-v2-base-multilingual
rerank_model = JinaRerank(
    api_key=config["JINA_API_KEY"],
    model=RERANK_MODEL,
    top_n=5
    )