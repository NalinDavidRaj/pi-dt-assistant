
import streamlit as st
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import pinecone
from langchain_cohere import ChatCohere
from langchain_cohere import CohereEmbeddings
from streamlit.logger import get_logger


COHERE_API_KEY =  "7Sc4f917kCqYSB3hyAYbsaCFLbFXZQUBucGKZjsw"
DB_FAISS_PATH = r"vectorstore/db_faiss"
embeddings = CohereEmbeddings(model="embed-english-light-v3.0")

st.title("ChatGPT-like clone")

db = FAISS.load_local(DB_FAISS_PATH, embeddings,allow_dangerous_deserialization=True)
llm = ChatCohere(model="command-r")
qa = ConversationalRetrievalChain.from_llm(llm,retriever=db.as_retriever())

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Adapted code for ConversationalRetrievalChain
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=db.as_retriever())
    query = prompt  # Use the user's input as the query
    chat_history = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
    result = qa({"question": query})
    #response = result["answer"]

    # Print the response
    #print("Response:", response)

    #with st.chat_message("assistant"):
        #st.write(response)

    #st.session_state.messages.append({"role": "assistant", "content": response})