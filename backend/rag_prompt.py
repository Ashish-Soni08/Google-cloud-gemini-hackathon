from llama_index.core import PromptTemplate

template = (
    """ You are an assistant for question-answering tasks.
Use the following context to answer the question.
If you don't know the answer, just say that you don't know.
Use five sentences maximum and keep the answer concise.\n
Question: {query_str} \nContext: {context_str} \nAnswer:"""
)

llm_prompt = PromptTemplate(template)