from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Together
import os
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
import time

st.set_page_config(page_title="JusticeBot")

# Layout for the logo
col1, col2, col3 = st.columns([1,4,1])
with col2:
    st.image("logo.png")

# Styling
st.markdown(
    """
    <style>
        .stApp, .ea3mdgi6 { background-color: #000000; }
        div.stButton > button:first-child { background-color: #ffd0d0; }
        div[data-testid="stStatusWidget"] div button { display: none; }
        .reportview-container { margin-top: -2em; }
        #MainMenu, .stDeployButton, footer, #stDecoration, button[title="View fullscreen"] { visibility: hidden; }
        button:first-child { background-color: transparent !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Reset conversation function
def reset_conversation():
    st.session_state["messages"] = []
    st.session_state["memory"].clear()

# Initialize session state for messages and memory
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferWindowMemory(
    k=2,
    memory_key="chat_history",
    chat_memory=ChatMessageHistory()
)

# Initialize embeddings and database retriever
embedings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"trust_remote_code": True, "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"}
)
db = FAISS.load_local("./vector_db", embedings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Set up the prompt template
prompt_template = """
<s>[INST]This is a chat template and as a legal chatbot specializing in Indian Labour Law queries, your primary objective is to provide accurate and concise information based on the user's questions. Do not generate your own questions and answers. You will adhere strictly to the instructions provided, offering relevant context from the knowledge base while avoiding unnecessary details. Your responses will be brief, to the point, and in compliance with the established format. If a question falls outside the given context, you will refrain from utilizing the chat history and instead rely on your own knowledge base to generate an appropriate response. You will prioritize the user's query and refrain from posing additional questions. The aim is to deliver professional, precise, and contextually relevant information pertaining to Indian Labour Laws.
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])

# Set up the LLM and conversational chain
llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.5,
    max_tokens=1024,
    api_key=os.getenv("TOGETHER_API_KEY")
)


qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=st.session_state["memory"],
    retriever=db_retriever,
    combine_docs_chain_kwargs={'prompt': prompt}
)

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Input prompt for new user message
input_prompt = st.chat_input("Ask Question")

# Handle user input and AI response
if input_prompt:
    with st.chat_message("user"):
        st.write(input_prompt)

    # Store the user message in session state
    st.session_state["messages"].append({"role": "user", "content": input_prompt})

    # Generate and display the assistant's response
    with st.chat_message("assistant"):
        with st.status("Thinking 💡...", expanded=True):
            result = qa.invoke(input=input_prompt)
            response = result["answer"]

            # Displaying the assistant's response
            full_response = "⚠️ **_Note: Information provided may be inaccurate._**\n\n\n" + response
            st.write(full_response)
            st.session_state["messages"].append({"role": "assistant", "content": full_response})

    # Reset button
    st.button("Reset All Chat 🗑️", on_click=reset_conversation)
