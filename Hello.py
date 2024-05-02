# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import pinecone
from langchain_cohere import ChatCohere
from langchain_cohere import CohereEmbeddings
import streamlit as st
from streamlit.logger import get_logger
import random
import time
import os
import sys
import getpass

LOGGER = get_logger(__name__)
os.environ["COHERE_API_KEY"] = getpass.getpass()
DB_FAISS_PATH = "vectorstore/db_faiss"

#Train with CSV files
def train_model_With_CSV():
    loader = CSVLoader(file_path="Data/sample.csv",encoding="utf8",csv_args={'delimiter':','})
    data=loader.load()
    #spilt the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    text_chunks = text_splitter.split_documents(data)
    #Downlaod the model for embedding
    embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
    #convert the text chunks into embedding and save into FAISS knowledge base
    docsearch  = FAISS.from_documents(text_chunks,embeddings)
    #save to vector Db
    docsearch.save_local(DB_FAISS_PATH)

   

# Streamed response emulator
def response_generator():
    response ="How can I assist you today?"
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

def chat():
    title = "Performance Insights - DT Assistant"
    st.set_page_config(
        page_title=title,
        page_icon="👋",
    )
    
    st.title(title)
    # Initialize chat history
    if "messages" not in st.session_state:
      st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
      with st.chat_message(message["role"]):
        st.markdown(message["content"])  
    
    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
          st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator())
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

if st.button('Check availability'):
   st.write('Why hello there')
   
if __name__ == "__main__":
    chat()
