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

import streamlit as st
from streamlit.logger import get_logger
import random
import time

LOGGER = get_logger(__name__)



def run():
    st.set_page_config(
        page_title="PIDT Assistant",
        page_icon="👋",
    )
    
    st.title("Simple chat")
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

if __name__ == "__main__":
    run()
