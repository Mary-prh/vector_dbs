"""
This script ingests a PDF document, splits it into manageable chunks, embeds the chunks using OpenAI embeddings, 
stores them in a FAISS vector store, and then allows the user to ask questions interactively. 
The script will retrieve relevant information from the document and provide answers until the user decides to exit.

"""

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings , OpenAI
from langchain_community.vectorstores import FAISS  # to have a local vectorstore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()


if __name__ == "__main__":
    pdf_path = 'MTCNN_paper.pdf'
    loader = PyPDFLoader(pdf_path)
    document = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n") # no overlap between the chunks
    text = text_splitter.split_documents(document)
    print(f"Created {len(text)} chunks")

    embeddings = OpenAIEmbeddings()
    llm = OpenAI()

    vectorstore = FAISS.from_documents(text, embeddings)
    vectorstore.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local(
        "faiss_index_react", embeddings, allow_dangerous_deserialization=True
    )

    retreival_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combine_docs_chain = create_stuff_documents_chain(llm=llm , prompt=retreival_qa_chat_prompt)# Create a chain for passing a list of Documents to a model
    
    retreival_chain = create_retrieval_chain(retriever=new_vectorstore.as_retriever(),
                                             combine_docs_chain=combine_docs_chain)
    
    while True:
        user_query = input(f"Ask a question about the {pdf_path} article (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            print("Exiting...")
            break

        res = retreival_chain.invoke(input={"input": user_query})
        print(res["answer"])