"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle
import os

from langchain.document_loaders import ReadTheDocsLoader
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

doc_dir = "./doc"


def ingest_pdf():
    # Initialize an empty list to hold all the documents
    docs = []
    # Loop through all PDF files in the `doc` directory
    for file_name in os.listdir(doc_dir):
        if file_name.endswith(".pdf"):
            # Load the PDF file
            pdf_path = os.path.join(doc_dir, file_name)
            pdf = PyPDFLoader(file_path=pdf_path).load()
            # Split the PDF into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            chunks = text_splitter.split_documents(pdf)
            # Add the chunks to the list of all documents
            docs.extend(chunks)
    # Create an OpenAIEmbeddings object
    embeddings = OpenAIEmbeddings()
    # Create a FAISS vectorstore from the list of documents
    vectorstore = FAISS.from_documents(docs, embeddings)
    # Save the vectorstore to a file
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


def ingest_docs():
    """Get documents from web pages."""
    loader = ReadTheDocsLoader("langchain.readthedocs.io/en/latest/")
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    # ingest_docs()
    ingest_pdf()
