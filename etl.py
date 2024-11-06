from typing import List

from dotenv import dotenv_values

from llama_index.core import SimpleDirectoryReader

from llama_index.core.schema import (BaseNode,
                                     Document,
                                     MetadataMode)

from llama_parse import LlamaParse

config = dotenv_values(".env")

def extract(pdf_document: List[str] = ["sample_data/ozempic.pdf"], language: str = "en", target_pages: str = None):

    parsing_instructions = """
    The provided document is a thin piece of folded paper that is part of every drug prescription box. 
    Usually the text is in VERY small print and typically provides information about dosages, side effects, storage instructions and much more. 
    Try to extract the key information so that it is easy to understand.
    """

    pdf_parser = LlamaParse(
    api_key=config["LLAMACLOUD_API_KEY"],
    result_type="text",  # markdown doesn't work with fast_mode to True
    parsing_instruction=parsing_instructions,
    num_workers=7,
    check_interval=2,
    max_timeout=2000,
    verbose=True,
    show_progress=True,
    language=language,
    invalidate_cache=False,
    do_not_cache=False,
    fast_mode=True, # fast_mode=True doesn't work with result_type="markdown"
    ignore_errors=True,
    split_by_page=True,
    disable_ocr=True,
    target_pages=target_pages  # for testing purposes use target_pages="0,80" to only parse the first and last page 
    )

    file_extractor = {".pdf": pdf_parser}

    documents = SimpleDirectoryReader(input_files=pdf_document, 
                                      file_extractor=file_extractor,
                                      filename_as_id=True,
                                      required_exts=[".pdf"],
                                      num_files_limit=1).load_data()
    
    return documents


def add_metadata_to_documents(documents: List[Document]) -> List[Document]:
    for document in documents:
        original_metadata = document.metadata

        additional_metadata = {
            "total_pages_in_original_pdf": len(documents),
            "size_of_original_pdf(MB)": f"{original_metadata.get('file_size') / (1024*1024):.2f} MB"
        }

        document.metadata = original_metadata | additional_metadata
    
    return documents

def transform(documents: List[Document]) -> List[Document]:
    transformed_documents = []
    for document in documents:
        transformed_documents.append(
            Document(
                text=document.text,
                metadata=document.metadata,
                excluded_llm_metadata_keys=["file_name", "file_path", "file_type", "file_size", "creation_date", "last_modified_date", "total_pages_in_original_pdf", "size_of_original_pdf(MB)"],
                excluded_embed_metadata_keys = ["file_path", "file_type", "file_size", "creation_date", "last_modified_date", "total_pages_in_original_pdf", "size_of_original_pdf(MB)"],
                metadata_seperator="::",
                metadata_template="{key}=>{value}",
                text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
            )
        )
    return transformed_documents
