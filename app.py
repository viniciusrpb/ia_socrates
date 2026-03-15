import os
import json
import streamlit as st
import uuid
import re
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
    current_chapter = "unknown"

    arquivos = []

    for f in os.listdir(fpath):
        if f.endswith(".json"):
            arquivos.append(f)

    for filename in arquivos:

        full_path = os.path.join(fpath, filename)

        with open(full_path, "r", encoding="utf-8") as f:

            data = json.load(f)

            paragrafos = data.get("paragrafos", [])

            for p in paragrafos:

                texto = p.get("texto", "").strip()

                if texto == "":
                    continue

                chap = re.match(r'^\d+(\.\d+)+', texto)

                if chap:
                    current_chapter = chap.group()

                section_id = str(uuid.uuid4())

                doc = Document(page_content=texto,
                    metadata={
                        "section_id": section_id,
                        "chapter": current_chapter,
                        "fonte": filename,
                        "pagina": p.get("pagina"),
                        "habilidades": p.get("habilidades", [])
                    }
                )

                sections.append(doc)

    return sections

@st.cache_resource(show_spinner=True)
def setup_hierarchical_retriever():

    sections = load_jsons(source_dir)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)

    chunks = []

    for section in sections:

        section_id = section.metadata["section_id"]
        chapter = section.metadata["chapter"]
        skills = section.metadata.get("habilidades", [])

        split_docs = splitter.split_documents([section])

        for c in split_docs:

            c.metadata["section_id"] = section_id
            c.metadata["chapter"] = chapter
            c.metadata["habilidades"] = skills

            chunks.append(c)

    embeddings = HuggingFaceEmbeddings(model_name="neuralmind/bert-base-portuguese-cased")

    chapter_docs = []
    seen = set()

    for s in sections:

        ch = s.metadata["chapter"]

        if ch not in seen:

            doc = Document(page_content=ch,metadata={"chapter": ch})

            chapter_docs.append(doc)

            seen.add(ch)

    chapter_index = FAISS.from_documents(chapter_docs, embeddings)
    section_index = FAISS.from_documents(sections, embeddings)
    chunk_index = FAISS.from_documents(chunks, embeddings)

    return chapter_index, section_index, chunk_index


def hierarchicalRetrieve(query):

    chapter_docs = chapter_index.similarity_search(query, k=3)

    chapters = set()

    for d in chapter_docs:
        chapters.add(d.metadata["chapter"])

    section_docs = section_index.similarity_search(query, k=10)

    section_ids = set()

    for d in section_docs:
        chapter = d.metadata["chapter"]
        if chapter in chapters:
            section_ids.add(d.metadata["section_id"])

    chunk_docs = chunk_index.similarity_search(query, k=80)

    filtered = []

    for c in chunk_docs:
        sid = c.metadata["section_id"]
        if sid in section_ids:
            filtered.append(c)

    return filtered[:N]

N = 8

source_dir = "knowledgeBase"

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
