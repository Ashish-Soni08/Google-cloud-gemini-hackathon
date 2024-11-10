# PillPal💊

![PillPal_Logo](PillPal_logo.png)

## Problem Statement

Have you ever come across the thin piece of folded paper that is part of every drug prescription box. Usually the text is in very small print and typically provides information about dosages, side effects, storage instructions and much more. They are hard to read and understand and requires some effort to get answers to common questions that you as a patient might have.

## Solution

Create a product that answers these questions and actually makes the medical information more accessible and easier to understand - enter PillPal💊

A Retrieval Augmented Generation (RAG) based chatbot that answers questions based on the PDF document.

## Environement Setup

```bash
python -V
# Output: Python 3.12.1
```

```bash
# create a environment named -> google-ai
python -m venv google-ai
```

```bash
# activate the environment
source google-ai/bin/activate
```

```bash
# create a Jupyter Notebook kernel
pip install jupyter ipykernel
```

```bash
# add the virtual environment as a kernel for the jupyter notebook
python -m ipykernel install --user --name=google-ai --display-name="Py3.12-google-ai"
```

```bash
# verify kernel installation
jupyter kernelspec list
```

## ARCHITETURE OF THE APPLICATION

### DATA SOURCES

- [**DIRECTORY READER(PDFs)**](https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/)

### MODEL PROVIDERS

#### GOOGLE

- **Model ID:** `gemini-1.5-pro`
- **Developed by:** Google
- [**Model Card**](https://ai.google.dev/gemini-api/docs/models/gemini#gemini-1.5-pro)
- [**LlamaIndex Docs**](https://docs.llamaindex.ai/en/stable/api_reference/llms/gemini/)
- [**LLamaIndex Example**](https://docs.llamaindex.ai/en/stable/examples/llm/gemini/)

#### EMBEDDING MODEL

- **Model ID:** `text-embedding-004`
- **Developed by:** Google
- [**Model Card**](https://ai.google.dev/gemini-api/docs/models/gemini#text-embedding-and-embedding)
- [**LlamaIndex Docs**](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/embeddings/llama-index-embeddings-gemini/llama_index/embeddings/gemini/base.py)
- - [**LLamaIndex Example**](https://docs.llamaindex.ai/en/stable/examples/embeddings/gemini/)
  
#### GROQ

- **Model ID:** `llama-guard-3-8b`
- **Developed by:** `Meta`
- **Context Window:** `8,192 tokens`
- [**Model Card**](https://huggingface.co/meta-llama/Llama-Guard-3-8B)
- [**GROQ Docs**](https://console.groq.com/docs/content-moderation)

## Built for

[Google Cloud Gemini Hackathon](https://googlecloudgeminihackathon.devpost.com/?ref_content=default&ref_feature=challenge&ref_medium=portfolio)

## Resources

[Ideal Chunk Size of a RAG System](https://www.llamaindex.ai/blog/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5)

[Metadata Customization](https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_documents/)