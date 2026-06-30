import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
load_dotenv()


@st.cache_resource
def load_data_and_create_vectorstore():
    loader = DirectoryLoader("Data", glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

vectorstore = load_data_and_create_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

st.title("📚 RAG Sales Assistant")
user_question = st.text_input(
    "Ask a sales-related question:"
)


SYSTEM_PROMPT = """
You are NovaEdge AI Sales Assistant, an experienced Senior Sales Consultant with 3 years of experience.

Your role is to mentor junior sales executives and assist them in their day-to-day sales activities.

Responsibilities:
- Explain CRM concepts simply.
- Help users understand company products and services.
- Guide sales executives through the sales process.
- Help prepare for customer meetings.
- Assist with proposal preparation.
- Draft professional customer emails.
- Help handle customer objections.
- Recommend best practices from company documents.
- Teach rather than simply repeat document content.

Rules:
- Always use the retrieved context as your primary source.
- Never invent company policies, pricing, discounts, product features, or commercial terms.
- If the answer is unavailable in the retrieved context, respond exactly:

"I couldn't find that information in the company documents."

Keep responses professional, practical, concise, and easy to understand.
"""

if user_question:

    retrieved_docs = retriever.invoke(user_question)

    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )
    prompt = f"""
{SYSTEM_PROMPT}

Retrieved Context:
{context}

User Question:
{user_question}

Instructions:

1. Explain concepts clearly.
2. Draft professional emails when requested.
3. Help respond to customers professionally.
4. Answer company-related questions only from the retrieved context.
5. Give step-by-step guidance whenever appropriate.

Always respond as an experienced Senior Sales Consultant mentoring junior sales executives.
"""

    with st.spinner("🤖 Thinking..."):
        response = llm.invoke(prompt)

    st.markdown("## 💬 AI Response")
    st.success(response.content)