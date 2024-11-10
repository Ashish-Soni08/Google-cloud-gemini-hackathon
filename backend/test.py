from dotenv import dotenv_values

import google.generativeai as genai

from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

config = dotenv_values(".env")

# genai.configure(api_key=config["GEMINI_API_KEY"])

LLM_MODEL: str = "models/gemini-1.5-pro"

llm = Gemini(api_key=config["GEMINI_API_KEY"],
             model=LLM_MODEL,
             temperature=0.2)

print(llm.complete("What is the capital of France?"))

EMBEDDING_MODEL: str = "models/text-embedding-004"

# Gemini Embedding
embed_model = GeminiEmbedding(api_key=config["GEMINI_API_KEY"],
                                  model=EMBEDDING_MODEL,
                                  truncate="END")

embeddings = embed_model.get_text_embedding("Google Gemini Embeddings.")
print(f"Dimension of embeddings: {len(embeddings)}")