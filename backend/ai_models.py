from dotenv import dotenv_values

import google.generativeai as genai

from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

config = dotenv_values(".env")

# MODELS
EMBEDDING_MODEL: str = "models/text-embedding-004"

LLM_MODEL: str = "models/gemini-1.5-pro"

# text-embedding-004
embedding_model = GeminiEmbedding(api_key=config["GEMINI_API_KEY"],
                                  model=EMBEDDING_MODEL,
                                  truncate="END")

# gemini-1.5-pro
llm = Gemini(api_key=config["GEMINI_API_KEY"],
             model=LLM_MODEL,
             temperature=0.2)

# llama-guard-3-8b
GUARDRAIL_MODEL: str = "llama-guard-3-8b"