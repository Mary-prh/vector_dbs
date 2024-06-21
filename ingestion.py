"""
Implement langchain code to ingest our text into the Pinecone vector store
"""
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
load_dotenv()


if __name__ == "__main__":

    loader = TextLoader('mediumblog1.txt', encoding = 'UTF-8')
    document = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0) # no overlap between the chunks
    text = text_splitter.split_documents(document)
    print(f"Created {len(text)} chunks")

    print("Embedding")
    embeddings = OpenAIEmbeddings(openai_api_type=os.environ.get('OPENAI_API_KEY')) # It uses the Open AI API to embed our documents

    print("Ingesting...")
    PineconeVectorStore.from_documents(documents=text, embedding=embeddings, index_name=os.environ["INDEX_NAME"])
    print("Finish!")