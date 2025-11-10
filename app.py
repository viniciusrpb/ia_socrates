import os
import json
import streamlit as st
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

st.set_page_config(page_title="IAssistente Sócrates - IAgora Brasil", layout="centered")
st.title("IAssistente Sócrates - IAgora Brasil")

client = Groq(api_key=st.secrets["GROQ_API"] or os.getenv("GROQ_API"))

def flatten_hierarchy(obj, path=""):

    chunks = []
    if isinstance(obj, dict):
        current_title = obj.get("título") or obj.get("capítulo") or path
        new_path = f"{path} > {current_title}" if path else current_title

        if "parágrafos" in obj:
            texto = " ".join(obj["parágrafos"])
            chunks.append({"titulo": new_path, "texto": texto})

        for key in ["seções", "subseções", "subsubseções", "capítulos"]:
            if key in obj:
                for sub in obj[key]:
                    chunks.extend(flatten_hierarchy(sub, new_path))
    elif isinstance(obj, list):
        for item in obj:
            chunks.extend(flatten_hierarchy(item, path))
    return chunks


def flatten_hierarchy(obj, path=""):
    chunks = []
    if isinstance(obj, dict):
        current_title = obj.get("título") or obj.get("capítulo") or path
        new_path = f"{path} > {current_title}" if path else current_title

        if "parágrafos" in obj:
            texto = " ".join(obj["parágrafos"])
            chunks.append({"titulo": new_path, "texto": texto})

        for key in ["seções", "subseções", "subsubseções", "capítulos"]:
            if key in obj:
                for sub in obj[key]:
                    chunks.extend(flatten_hierarchy(sub, new_path))
    elif isinstance(obj, list):
        for item in obj:
            chunks.extend(flatten_hierarchy(item, path))
    return chunks


@st.cache_data(show_spinner=False)
def load_jsons(folder_path):

    documentos = []

    if not os.path.exists(folder_path):
        st.error(f" A pasta '{folder_path}' não existe.")
        return []

    arquivos = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    if not arquivos:
        st.warning(f" Nenhum arquivo .json encontrado em '{folder_path}'.")
        return []

    for filename in arquivos:
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                chunks = flatten_hierarchy(data)
                for c in chunks:
                    documentos.append(
                        Document(
                            page_content=c["texto"],
                            metadata={"titulo": c["titulo"], "fonte": filename},
                        )
                    )
            st.write(f"📄 {filename}: {len(chunks)} seções lidas.")
        except Exception as e:
            st.warning(f" Erro ao ler {filename}: {e}")

    return documentos

@st.cache_resource(show_spinner=True)
def setup_retriever():

    SOURCE_DIR = 'source'

    st.info("Indexando documentos da BNCC...")
    docs = load_jsons(SOURCE_DIR)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=70)
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    db = FAISS.from_documents(split_docs, embedding=embeddings)
    st.success(f"{len(split_docs)} chunks indexados a partir de {len(docs)} seções da BNCC.")
    return db.as_retriever(search_kwargs={"k": 5})

retriever = setup_retriever()

pergunta = st.text_input("Digite sua pergunta:")

if pergunta:
    with st.spinner("Buscando resposta..."):
        docs = retriever.invoke(pergunta)
        contexto = "\n\n".join(
            [f"[{doc.metadata['titulo']}]\n{doc.page_content}" for doc in docs]
        )

        prompt = f"""
Você é um assistente educacional especializado em currículos e diretrizes da BNCC.
Use apenas as informações fornecidas abaixo como base de conhecimento e destaque a hierarquia (capítulo, seção, subseção) quando apropriado.

Contexto:
{contexto}

Pergunta:
{pergunta}

Resposta:
"""

        response = client.chat.completions.create(
            model="llama3-70b",
            messages=[
                {"role": "system", "content": "Você é um assistente confiável que responde em português, com precisão e clareza."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=512,
        )

        resposta = response.choices[0].message.content

        st.subheader("Resposta:")
        st.write(resposta)

        with st.expander("Contextos utilizados"):
            for d in docs:
                st.markdown(f"**{d.metadata['titulo']}** — {d.metadata['fonte']}")
                st.caption(d.page_content[:400] + "...")

