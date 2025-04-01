from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain_huggingface.llms import HuggingFaceEndpoint
import tempfile
import streamlit as st
import os

# 🔹 Initialize Session State Variables
if "mode" not in st.session_state:
    st.session_state.mode = "login"

if "documents" not in st.session_state:
    st.session_state.documents = []

if "hf_api_key" not in st.session_state:
    st.session_state.hf_api_key = ""

if "pinecone_api_key" not in st.session_state:
    st.session_state.pinecone_api_key = ""

if "pinecone_env" not in st.session_state:
    st.session_state.pinecone_env = ""

if "pinecone_index_name" not in st.session_state:
    st.session_state.pinecone_index_name = ""

# 🔹 Ensure Chat History is Initialized
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("RAG With Pinecone and Langchain")

if st.session_state.mode == "login":
    st.session_state.pinecone_api_key = st.text_input("Pinecone API Key")
    st.session_state.pinecone_env = st.text_input("Pinecone Environment")
    st.session_state.pinecone_index_name = st.text_input("Pinecone Index Name")
    "---"
    st.session_state.hf_api_key = st.text_input("Hugging Face Access Token")
    if st.button("Continue"):
        if st.session_state.hf_api_key is not None:
            if st.session_state.pinecone_api_key is not None:
                if st.session_state.pinecone_env is not None:
                    if st.session_state.pinecone_index_name is not None:
                        if "embeddings" not in st.session_state:
                            st.session_state.embeddings = HuggingFaceInferenceAPIEmbeddings(
                                api_key=st.session_state.hf_api_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
                            )

                        if "vector_store" not in st.session_state:
                            st.session_state.vector_store = PineconeVectorStore(
                                index_name=st.session_state.pinecone_index_name,
                                embedding=st.session_state.embeddings,
                                pinecone_api_key=st.session_state.pinecone_api_key
                            )

                        st.session_state.mode = "input"
                        st.rerun()


# 🔹 PDF Upload & Processing
if st.session_state.mode == "input":
    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type="pdf")

    if st.button("Process PDFs"):
        if uploaded_files:
            with tempfile.TemporaryDirectory() as temp_dir:
                for uploaded_file in uploaded_files:
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.read())

                    loader = PyPDFLoader(temp_path)
                    pages = loader.load()

                    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
                    split_docs = splitter.split_documents(pages)
                    st.session_state.documents.extend(split_docs)

            # 🔹 Add documents to Vector Store
            st.session_state.vector_store.add_documents(st.session_state.documents)
            st.session_state.mode = "chat"
            st.rerun()


# 🔹 Chat Mode
if st.session_state.mode == "chat":
    prompt = st.chat_input("Ask with reference to document")

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)

        retriever = st.session_state.vector_store.as_retriever()
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            temperature=0.8,
            huggingfacehub_api_token=st.session_state.hf_api_key
        )

        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

        response = retrieval_chain.invoke({"input": prompt})

        # 🔹 Extract Context and Answer
        answer = response.get("answer", "No answer found.")
        context_docs = response.get("context", [])
        context_texts = [doc.page_content for doc in context_docs]

        # 🔹 Save to Chat History
        st.session_state.chat_history.append({
            "question": prompt,
            "answer": answer,
            "context": context_texts
        })

        with st.chat_message("ai"):
            st.markdown(f"**AI's Response:** {answer}")

            # 🔹 Show Document Context
            if context_texts:
                st.markdown("**Context from Documents:**")
                for i, doc_text in enumerate(context_texts):
                    with st.expander(f"📖 View Context {i+1}"):
                        st.markdown(doc_text)

    # 🔹 Display Chat History
    st.divider()
    st.markdown("### Chat History")

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat.get("question", "No question recorded."))

        with st.chat_message("ai"):
            st.markdown(f"**AI's Response:** {chat.get('answer', 'No answer found.')}")

            if chat.get("context"):
                st.markdown("**Context from Documents:**")
                for i, doc_text in enumerate(chat["context"]):
                    with st.expander(f"📖 View Context {i+1}"):
                        st.markdown(doc_text)
