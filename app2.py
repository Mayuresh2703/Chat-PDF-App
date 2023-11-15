import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain

with st.sidebar:
    st.title('PDF Based Chatbot')
    st.markdown("## Conversation History: ")

    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}

    # New Chat button
    if "active_session" not in st.session_state or st.sidebar.button("New Chat +", use_container_width=True):
        # Create a new chat session and set it as active
        chat_id = len(st.session_state.chat_sessions) + 1
        session_key = f"Chat {chat_id}"
        st.session_state.chat_sessions[session_key] = []
        st.session_state.active_session = session_key

    # Buttons for previous chat sessions
    for session in st.session_state.chat_sessions:
        if st.sidebar.button(session, key=session):
            st.session_state.active_session = session
    st.markdown('''
    ## About App:

    This app is an LLM powered chatbot  using:

    - [Streamlit](https://streamlit.io/)
    - [Langchain](https://docs.langchain.com/docs/)
    - [OpenAI](https://openai.com/)

    ## About me:

    - [Github](https://github.com/Mayuresh2703)

    ''')
    st.write("Made by [Mayuresh]")

def main():
    api_key = st.text_input("Enter your OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

        st.header("Chat with your PDF File")

        # PDF upload and processing
        pdf = st.file_uploader("Upload your PDF:", type='pdf')

        # extract the text
        if pdf is not None:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            # split text into chunks
            text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(text)

            # Open AI embeddings and vector store 
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)

            # Open AI LLM and initializing a Conversation Chain using langchain
            llm = OpenAI(temperature=0)
            qa_chain = ConversationalRetrievalChain.from_llm(llm, vector_store.as_retriever())

            if "active_session" in st.session_state:
                for message in st.session_state.chat_sessions[st.session_state.active_session]:
                    role = message["role"]
                    content = message["content"]
                    st.text(f"{role}: {content}")

            # Read user input prompt
            query = st.text_input("Ask your questions from PDF ")

            if query:
                # using chat message to initiate User conversation
                st.session_state.chat_sessions[st.session_state.active_session].append({"role": "user", "content": query})
                st.text(f"user: {query}")

                # Generate response using qa chain with the help of query and previous messages
                result = qa_chain({"question": query, "chat_history": [(message["role"], message["content"]) for message in st.session_state.chat_sessions[st.session_state.active_session]]})
                response = result["answer"]

                # using chat message to initiate Bot conversation
                st.session_state.chat_sessions[st.session_state.active_session].append({"role": "assistant", "content": response})
                st.text(f"assistant: {response}")

if __name__ == '__main__':
    main()
