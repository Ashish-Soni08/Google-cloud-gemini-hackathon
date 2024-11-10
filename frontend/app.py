from dotenv import dotenv_values

import gradio as gr
from gradio_pdf import PDF

from llama_index.core import (load_index_from_storage,
                              StorageContext,
                              Settings,
                              VectorStoreIndex
                              )
from llama_index.core.node_parser import SentenceSplitter

from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from backend import (add_metadata_to_documents,
                     extract,
                     transform,
                     embedding_model,
                     llm,
                     moderate_message,
                     llm_prompt)

config = dotenv_values("backend/.env")

# Configure settings for the application
Settings.embed_model = embedding_model
Settings.llm = llm
Settings.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=50)


qdrant_client = QdrantClient(url=config["QDRANT_ENDPOINT"], 
                             api_key=config["QDRANT_API_KEY"])

# Initialize global variables for the index and query engine
index = None
query_engine = None

# Function to load documents and create the index
def load(document: str = None, progress: gr.Progress = gr.Progress()) -> None:
    global index, query_engine
    try:
            
        documents = transform(add_metadata_to_documents(extract(document)))

        # Create a Qdrant vector store and storage context
        vector_store = QdrantVectorStore(client=qdrant_client,
                                         collection_name="pillpal_documents",
                                         )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create the index from the documents
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        # Create the query engine
        query_engine = index.as_query_engine(similarity_top_k=5, streaming=True)
        
        return f"Successfully loaded {len(documents)} documents into Qdrant Vector Store."
    except Exception as e:
        return f"Error loading documents: {str(e)}"


# Function to handle chat interactions
def chat(message, history):
    global query_engine
    if query_engine is None:
        return history + [("Please load documents first.", None)]
    try:
        response = query_engine.query(message)
        return history + [(message, response)]
    except Exception as e:
        return history + [(message, f"Error processing query: {str(e)}")]

# Function to stream responses
def stream_response(message, history):
    global query_engine
    if query_engine is None:
        yield history + [("Please load documents first.", None)]
        return
    
    try:
        response = query_engine.query(message)
        partial_response = ""
        for text in response.response_gen:
            partial_response += text
            yield history + [(message, partial_response)]
    except Exception as e:
        yield history + [(message, f"Error processing query: {str(e)}")]

####### Gradio Application #######

theme = gr.themes.Soft()

title = """<h1 id="title"> PillPal💊</h1>"""

description = """Chatbot for drug information. Upload a PDF document to get started"""

css = """h1#title {
    text-align: center;
    } 
"""

pillpal_bot = gr.Blocks(theme=theme, css=css, fill_height=True, fill_width=True)

with pillpal_bot:
    gr.Markdown(title)
    gr.Markdown(description)
    with gr.Row():
        with gr.Column(scale=2):
            pdf = PDF(label="Upload a PDF", interactive=True)
        with gr.Column(scale=2):
            chatbot = gr.ChatInterface(fn=stream_response,
                                        type="messages",
                                        chatbot=gr.Chatbot(label="Chat with PillPal",
                                                           type="messages",
                                                           inputs=[pdf],
                                                           show_share_button=True,
                                                           show_copy_button=True,
                                                           show_copy_all_button=True,
                                                           scale=2)
                                        )

if __name__ == "__main__":
    pillpal_bot.launch(debug=True)