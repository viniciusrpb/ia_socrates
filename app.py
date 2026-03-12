import os
import json
import streamlit as st
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

st.set_page_config(page_title="IAssistente Sócrates - IAgora Brasil", layout="centered")
st.title("IAssistente Sócrates - IAgora Brasil")

client = Groq(api_key=st.secrets["GROQ_API"] or os.getenv("GROQ_API"))

@st.cache_data(show_spinner=False)
def load_jsons(fpath):

    sections = []

    arquivos = [f for f in os.listdir(fpath) if f.endswith(".json")]

    for filename in arquivos:

        with open(os.path.join(fpath, filename), "r", encoding="utf-8") as f:

            data = json.load(f)

            for p in data.get("paragrafos", []):

                texto = p.get("texto", "").strip()

                if not texto:
                    continue

                section_id = str(uuid.uuid4())

                sections.append(Document(
                    page_content=texto,
                    metadata={
                        "section_id": section_id,
                        "fonte": filename,
                        "pagina": p.get("pagina"),
                        "habilidades": p.get("habilidades", [])
                    }
                ))

    return sections

@st.cache_resource(show_spinner=True)
def setup_hierarchical_retriever():

    SOURCE_DIR = "source"

    sections = load_jsons(SOURCE_DIR)

    splitter = RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=120)

    chunks = []

    for section in sections:

        section_id = section.metadata["section_id"]

        split_docs = splitter.split_documents([section])

        for c in split_docs:

            c.metadata["section_id"] = section_id
            chunks.append(c)

    embeddings = HuggingFaceEmbeddings(model_name="neuralmind/bert-base-portuguese-cased")

    section_index = FAISS.from_documents(sections, embeddings)

    chunk_index = FAISS.from_documents(chunks, embeddings)

    return section_index, chunk_index


def hierarchicalRetrieve(query):

    section_docs = section_index.similarity_search(query, k=5)

    temp = []
    for d in section_docs:
        temp.append(d.metadata["section_id"])

    section_ids = set(temp)

    chunk_docs = chunk_index.similarity_search(query, k=25)

    filtered = []
    for c in chunk_docs:
        if c.metadata["section_id"] in section_ids:
            filtered.append(c)

    return filtered[:8]


section_index, chunk_index = setup_hierarchical_retriever()

pergunta = st.text_input("Digite sua pergunta:")

if pergunta:
    with st.spinner("Buscando resposta..."):

        docs = hierarchicalRetrieve(pergunta)
        contexto = "\n\n".join([f"[{doc.metadata.get('fonte','?')} - pág. {doc.metadata.get('pagina','?')}]\n{doc.page_content}" for doc in docs])

        prompt = f"""
            Você é um assistente educacional que responde com base em documentos da BNCC, da BNCC na Computação e da Educação no Brasil. Use o seguinte contexto para responder com precisão à pergunta.

            Contexto:
            {contexto}

            Pergunta:
            {pergunta}

            Resposta:"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Você é um assistente confiável que responde em português, com precisão e clareza."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=512
        )

        resposta = response.choices[0].message.content

        st.subheader("Resposta:")
        st.write(resposta)
