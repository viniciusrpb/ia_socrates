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

def flatten_hierarchy(obj, path="", pagina_atual=None):
    chunks = []

    if isinstance(obj, dict):

        titulo_atual = obj.get("título") or obj.get("capítulo") or obj.get("nome") or path
        caminho = f"{path} > {titulo_atual}" if path else titulo_atual

        pagina = obj.get("página", pagina_atual)

        textos = []

        if "parágrafos" in obj:
            textos.extend(obj["parágrafos"])

        if "descrição" in obj:
            textos.append(obj["descrição"])

        if "habilidades" in obj:
            for h in obj["habilidades"]:
                textos.append(h.get("descrição") or h.get("texto") or "")

        if "direitos" in obj:
            textos.extend(obj["direitos"])

        if textos:
            chunks.append({
                "titulo": caminho,
                "texto": " ".join(textos),
                "pagina": pagina
            })

        for chave in ["capítulos", "seções", "subseções", "subsubseções", "eixos"]:
            if chave in obj:
                for sub in obj[chave]:
                    chunks.extend(flatten_hierarchy(sub, caminho, pagina))

    elif isinstance(obj, list):
        for item in obj:
            chunks.extend(flatten_hierarchy(item, path, pagina_atual))

    return chunks


@st.cache_data(show_spinner=False)
def load_jsons(fpath):

    documentos = []

    arquivos = [f for f in os.listdir(fpath) if f.endswith(".json")]

    for filename in arquivos:

        with open(os.path.join(fpath, filename), "r", encoding="utf-8") as f:

            data = json.load(f)

            for p in data.get("paragrafos", []):

                texto = p.get("texto", "").strip()

                if not texto:
                    continue

                documentos.append(
                    Document(
                        page_content=texto,
                        metadata={
                            "fonte": filename,
                            "pagina": p.get("pagina"),
                            "habilidades": p.get("habilidades", [])
                        },
                    )
                )

    return documentos

@st.cache_resource(show_spinner=True)
def setup_retriever():

    SOURCE_DIR = 'source'

    docs = load_jsons(SOURCE_DIR)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="neuralmind/bert-base-portuguese-cased" # atualizei aqui par ao BERTimbau, mas fiquem a vontade para alterar
    )

    db = FAISS.from_documents(split_docs, embedding=embeddings)
    #st.success(f"{len(split_docs)} chunks indexados a partir de {len(docs)} seções da BNCC.")
    return db.as_retriever(search_kwargs={"k": 7})

retriever = setup_retriever()

pergunta = st.text_input("Digite sua pergunta:")

if pergunta:
    with st.spinner("Buscando resposta..."):

        docs = retriever.invoke(pergunta)
        contexto = "\n\n".join(
            [f"[{doc.metadata['titulo']}]\n{doc.page_content}" for doc in docs]
        )

        prompt = f""" Você é um assistente educacional especializado em currículos e diretrizes da BNCC. Use somente o contexto fornecido abaixo. O contexto está organizado de forma hierárquica, contendo níveis como Título, Capítulo, Seção e Subseção. Ao responder, utilize a hierarquia para localizar e justificar a informação. Quando possível, mencione explicitamente de quais níveis hierárquicos a resposta foi derivada. Não invente informações que não estejam presentes no contexto. Se o contexto for insuficiente, diga claramente que não é possível responder. Forneça a fonte ao final da resposta.

        Contexto:
        {contexto}

        Pergunta:
        {pergunta}

        Resposta:
        """

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Você é um assistente confiável que responde em português, com precisão e clareza."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=512,
        )

        resposta = response.choices[0].message.content

        st.subheader("Resposta:")
        st.write(resposta)
