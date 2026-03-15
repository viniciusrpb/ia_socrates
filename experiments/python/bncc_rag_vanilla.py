from huggingface_hub import login
import pandas as pd
import torch
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    pipeline,
    AutoModelForSequenceClassification
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from dataclasses import dataclass
from datasets import Dataset
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
from bert_score import score as bertscore
import numpy as np

import os
import json
import torch
import pandas as pd
import numpy as np

from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bertscore

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from bleurt_pytorch import BleurtTokenizer, BleurtForSequenceClassification

from sentence_transformers import SentenceTransformer

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import AnswerCorrectness

from groq import Groq
from openai import AsyncOpenAI
import time

def compute_bleu(ref, hyp):
    smoothie = SmoothingFunction().method1
    return sentence_bleu([ref.split()], hyp.split(), smoothing_function=smoothie)

def compute_bertscore(reference, candidate):

    P, R, F1 = bertscore(
        [candidate],
        [reference],
        model_type="xlm-roberta-large",
        lang="pt",
        verbose=False
    )

    return float(F1[0])

def compute_bleurt(ref, hyp):
    inputs = bleurt_tokenizer(
        [ref],
        [hyp],
        padding='longest',
        return_tensors='pt',
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = bleurt_model(**inputs)
        score = outputs.logits.flatten().item()

    return float(score)

def ragas(question, reference, llmanswer):
    # fonte https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/answer_correctness/?h=correc#answer-correctness

    scorer = AnswerCorrectness(llm=llm, embeddings=embeddings)

    result = asyncio.run(
        scorer.ascore(
            user_input=question,
            response=llmanswer,
            reference=reference
        )
    )

    return float(result.value)

def compute_entailment(ref, hyp):
    inputs = nli_tokenizer(ref,hyp,return_tensors="pt", truncation=True,max_length=512).to(device)

    with torch.no_grad():
        outputs = nli_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)

    entailment_prob = probs[0][2].item()
    contradiction_prob = probs[0][0].item()

    return entailment_prob, contradiction_prob

def load_jsons(path):

    docs = []

    for file in os.listdir(path):

        if not file.endswith(".json"):
            continue

        full_path = os.path.join(path, file)

        with open(full_path, encoding="utf-8") as f:

            data = json.load(f)

        paragrafos = data.get("paragrafos", [])

        for p in paragrafos:

            texto = p.get("texto", "").strip()

            if texto == "":
                continue

            pagina = p.get("pagina")
            chapter = p.get("chapter")
            section = p.get("section")
            habilidades = p.get("habilidades", [])

            doc = Document(
                page_content=texto,
                metadata={
                    "fonte": file,
                    "pagina": pagina,
                    "chapter": chapter,
                    "section": section,
                    "habilidades": habilidades
                }
            )

            docs.append(doc)

    return docs

def setup_retriever():

    docs = load_jsons(source_dir)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="neuralmind/bert-base-portuguese-cased") #intfloat/multilingual-e5-large)

    db = FAISS.from_documents(split_docs, embedding=embeddings)

    return db.as_retriever(search_kwargs={"k": 8})

def rag_answer(question):

    docs = retriever.invoke(question)

    contexto = "\n\n".join([d.page_content for d in docs])

    prompt = f"""Você é um assistente educacional especializado na BNCC.
        Use apenas o contexto fornecido para responder.
        Responda em no máximo 3 frases curtas.
        Não inclua explicações adicionais.

        Contexto:
        {contexto}

        Pergunta:
        {question}

        Resposta:
    """

    response = groq_client.chat.completions.create(
        model=modelname,
        messages=[
            {"role":"user","content":prompt}
        ],
        temperature=0.1,
        max_tokens=150
    )

    return response.choices[0].message.content.strip()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

faq_dir = "eval/"
output_dir = "results/"
source_dir = "source/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

bleurt_model_name = "lucadiliello/BLEURT-20-D12"
bleurt_tokenizer = BleurtTokenizer.from_pretrained(bleurt_model_name)
bleurt_model = BleurtForSequenceClassification.from_pretrained(bleurt_model_name)
bleurt_model.to(device)
bleurt_model.eval()

nli_model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
nli_model.to(device)
nli_model.eval()

# RAgas with problems - <exception>    The output is incomplete due to a max_tokens length limit.</exception>
#llm = llm_factory("gpt-4o-mini", client=openai_client,max_tokens=2048)
#embeddings = embedding_factory("openai", model="text-embedding-3-small", client=openai_client)

retriever = setup_retriever()

for modelname in ["llama-3.1-8b-instant","llama-3.3-70b-versatile"]:

    modeloNome = "vanillaRAG_"+modelname

    rows = []

    for file in os.listdir(faq_dir):

        if not file.endswith(".tsv"):
            continue

        path = os.path.join(faq_dir, file)
        df = pd.read_csv(path, sep="\t")

        for _, row in df.iterrows():

            question = str(row["question"])
            ground_truth = str(row["answer"])

            model_answer = rag_answer(question)
            time.sleep(0.5)

            rows.append({
                "source_file": file,
                "question": question,
                "ground_truth": ground_truth,
                "model_answer": model_answer
            })

    print("Respostas geradas, indo agora para avaliacao")
    df_answers = pd.DataFrame(rows)

    results = []

    for _, row in df_answers.iterrows():

        question = str(row["question"])
        llmanswer = str(row["model_answer"])
        ground_truth = str(row["ground_truth"])

        bleu = compute_bleu(ground_truth, llmanswer)
        rougeL = scorer.score(ground_truth, llmanswer)["rougeL"].fmeasure
        bertscore_f1 = compute_bertscore(ground_truth, llmanswer)
        bleurt_score_val = compute_bleurt(ground_truth, model_answer)
        entailment_prob, contradiction_prob = compute_entailment(ground_truth, llmanswer)
        #ragas_answer_correctness = ragas(question,ground_truth, llmanswer)

        results.append({
                "question": question,
                "ground_truth": ground_truth,
                "answer": llmanswer,
                "bleu": bleu,
                "rougeL": rougeL,
                "bleurt": bleurt_score_val,
                "bertscore": bertscore_f1,
                "entailment_prob": entailment_prob,
                "contradiction_prob": contradiction_prob
                #"answer_correctness": ragas_answer_correctness,
            })

    out_df = pd.DataFrame(results)
    out_path = os.path.join(output_dir,file.replace(".tsv", f"{modeloNome}_results.csv"))

    out_df.to_csv(out_path, index=False)

    print("Ok.")
