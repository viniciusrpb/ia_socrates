import streamlit as st
import pandas as pd
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
import requests

st.set_page_config(page_title="IAssistente Sócrates", layout="centered")
st.title("IAssistente Sócrates")

client = Groq(api_key=st.secrets["GROQ_API"])

@st.cache_data(show_spinner=False)
def load_documents(csv_path="bncc.csv"):
    df = pd.read_csv(csv_path,sep=',')
    return [
        Document(page_content=row["context"], metadata={"question": row.get("question", "")})
        for _, row in df.iterrows()
    ]

@st.cache_resource(show_spinner=False)
def setup_retriever():
    docs = load_documents()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    db = FAISS.from_documents(chunks, embedding=embeddings)
    return db.as_retriever()

retriever = setup_retriever()

pergunta = st.text_input("Digite sua pergunta:")

if pergunta:
    with st.spinner("Buscando resposta... aguarde..."):
        documentos = retriever.invoke(pergunta)
        contexto = "\n".join([doc.page_content for doc in documentos])

        prompt = f"""
Você é um assistente educacional que responde com base em documentos da BNCC, da BNCC na Computação e da Educação no Brasil. Use o seguinte contexto para responder com precisão à pergunta.

Contexto:
{contexto}

Pergunta:
{pergunta}

Resposta:"""

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "Você é um assistente útil que responde em português."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=512
        )

        resposta = response.choices[0].message.content

        st.subheader("📌 Resposta:")
        st.write(resposta)
