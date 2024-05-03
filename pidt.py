from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import pinecone
from langchain_cohere import ChatCohere
from langchain_cohere import CohereEmbeddings
import streamlit as st
import os
import sys
import getpass
 
os.environ["COHERE_API_KEY"] ="7Sc4f917kCqYSB3hyAYbsaCFLbFXZQUBucGKZjsw" #st.secrets["COHERE_API_KEY"] 
#getpass.getpass()
 
DB_FAISS_PATH = "vectorstore/db_faiss"
# Read CSV files
#loader = CSVLoader(file_path="Data/california_tour_package.csv", encoding="utf8", csv_args={'delimiter': ','})
#data = loader.load()
# print(data)
 
loader = CSVLoader(file_path="Data/sample.csv", encoding="utf8", csv_args={'delimiter': ','})
data = loader.load()
 
#loader = PyPDFLoader("Data/Optum.pdf")
#data = loader.load_and_split()
 
# spilt the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)
# print(len(text_chunks))
 
# download sentence trasnformers from ebedding from hugging face
# embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
 
# convert the text chunks into embedding and save into FAISS knowledge base
docsearch = FAISS.from_documents(text_chunks, embeddings)
#print(docsearch.index.ntotal)
#read existing index
db = FAISS.load_local(DB_FAISS_PATH, embeddings,allow_dangerous_deserialization=True) 
#existingindex = db.index.ntotal
#print(db.index.ntotal)
docsearch.merge_from(db)
#aftertraining =docsearch.index.ntotal
#Save to Local path
docsearch.save_local(DB_FAISS_PATH)
db = FAISS.load_local(DB_FAISS_PATH, embeddings,allow_dangerous_deserialization=True) 
#print(f"Training completed Existing index : {existingindex} After Training : {aftertraining}")

#debug
query = "who owns Mark@demomail.com?"
results = db.similarity_search(query)
for doc in results:
    print(f"Content: {doc.page_content}, Metadata: {doc.metadata}")

"""
# calling llm model
llm = ChatCohere(model="command-r")
qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())
 
while True:
    chat_history = []
    query = input(f"Please enter your question : ")
    if query == 'exit':
        print("Exiting")
        sys.exit()
    if query == '':
        continue
    print("processing")
    result = qa({"question": query, "chat_history": chat_history})
    print("Response:", result['answer'])
"""