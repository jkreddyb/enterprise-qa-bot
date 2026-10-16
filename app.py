import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Enterprise Policy Q&A Bot", page_icon="🤖", layout="centered")

st.title("🏢 Enterprise Policy Q&A Assistant")
st.markdown("Ask questions about your enterprise documents, HR policies, and compliance manuals.")

# Initialize and cache the RAG bot so it doesn't reload on every interaction
@st.cache_resource
def initialize_bot():
    # 1. Create sample documents if they don't exist
    os.makedirs("./enterprise_docs", exist_ok=True)
    
    hr_path = "./enterprise_docs/hr_policy.txt"
    if not os.path.exists(hr_path):
        with open(hr_path, "w") as f:
            f.write("Company PTO Policy: Employees accrue 20 days of paid time off per year. Remote work is allowed up to 2 days per week with manager approval.")

    sec_path = "./enterprise_docs/security_manual.txt"
    if not os.path.exists(sec_path):
        with open(sec_path, "w") as f:
            f.write("Data Security Protocol: All company laptops must have disk encryption enabled. Personal devices are strictly prohibited from accessing internal production databases.")

    # 2. Load and chunk documents
    docs = []
    for file in os.listdir("./enterprise_docs"):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join("./enterprise_docs", file))
            docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # 3. Embeddings & Vector Store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 4. LLM & Chain
    llm = ChatOllama(model="llama3.2:3b", temperature=0)
    
    system_prompt = (
        "You are an expert enterprise policy assistant. Use the given context "
        "to answer the user's question accurately. If you do not know the answer "
        "based on the context, state clearly that the knowledge base cannot answer it.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

# Load the bot
rag_chain = initialize_bot()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input box at the bottom
if query := st.chat_input("Ask a policy question..."):
    # Add user message to state and display it
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate bot response
    with st.chat_message("assistant"):
        with st.spinner("Searching enterprise documents..."):
            response = rag_chain.invoke({"input": query})
            answer = response["answer"]
            st.markdown(answer)
            
    # Add assistant response to state
    st.session_state.messages.append({"role": "assistant", "content": answer})