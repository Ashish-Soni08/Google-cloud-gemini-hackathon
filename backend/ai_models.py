from dotenv import dotenv_values

import google.generativeai as genai

from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

config = dotenv_values(".env")

genai.configure(api_key=config["GEMINI_API_KEY"])

# MODELS HOSTED BY NVIDIA(NIMS) 
EMBEDDING_MODEL: str = "models/text-embedding-004"

LLM_MODEL: str = "models/gemini-pro"

RERANK_MODEL: str = "add model name"

# nv-embedqa-mistral-7b-v2
embedding_model = GeminiEmbedding(model=EMBEDDING_MODEL,
                                  truncate="END")

# llama-3.1-405b-instruct
llm = Gemini(api_key=config["GEMINI_API_KEY"],
             model=LLM_MODEL)


# nv-rerankqa-mistral-4b-v3
rerank_model = "add model"
