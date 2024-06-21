"""
Implementation of Retreival Augmented Generation:
- it takes the query, 
- embeds it and send it to the vectorstore to get the relevant documents,
- appends it to the prompts
- sends all those prompts to the llm

"""
import os
from dotenv import load_dotenv
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
load_dotenv()


if __name__ == "__main__":
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()

    query = "What is Pinecone in machine learning?"
    # this is to compare the answers
    # chain = PromptTemplate.from_template(query) | llm
    # result_1 = chain.invoke(input={})
    # print(result_1.content)

    vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)

    retreival_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combine_docs_chain = create_stuff_documents_chain(llm=llm , prompt=retreival_qa_chat_prompt)# Create a chain for passing a list of Documents to a model
    
    retreival_chain = create_retrieval_chain(retriever=vectorstore.as_retriever(),
                                             combine_docs_chain=combine_docs_chain)
    
    result_2 = retreival_chain.invoke(input={"input": query})
    print(result_2)