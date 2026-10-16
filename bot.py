import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

def initialize_bot():
    print("🚀 Initializing Enterprise Policy Q&A Bot...")

    # 1. Create sample enterprise documents folder and files if they don't exist
    os.makedirs("./enterprise_docs", exist_ok=True)
    
    hr_path = "./enterprise_docs/hr_policy.txt"
    if not os.path.exists(hr_path):
        with open(hr_path, "w") as f:
            f.write("Company PTO Policy: Employees accrue 20 days of paid time off per year. Remote work is allowed up to 2 days per week with manager approval.")

    sec_path = "./enterprise_docs/security_manual.txt"
    if not os.path.exists(sec_path):
        with open(sec_path, "w") as f:
            f.write("Data Security Protocol: All company laptops must have disk encryption enabled. Personal devices are strictly prohibited from accessing internal production databases.")

    # 2. Load Documents
    docs = []
    for file in os.listdir("./enterprise_docs"):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join("./enterprise_docs", file))
            docs.extend(loader.load())

    # 3. Chunk Documents (Chunk size 500, overlap 100)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # 4. Embed and store in local Chroma vector database
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 5. Set up LLM & Retrieval Chain
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

if __name__ == "__main__":
    bot = initialize_bot()

    # Simple interactive test loop right inside your VS Code terminal
    print("\n✅ Bot is ready! Type 'exit' to quit.\n")
    while True:
        query = input("Ask a question about your enterprise policies: ")
        if query.lower() in ["exit", "quit"]:
            break
        
        response = bot.invoke({"input": query})
        print(f"\nAnswer:\n{response['answer']}\n" + "-"*50)