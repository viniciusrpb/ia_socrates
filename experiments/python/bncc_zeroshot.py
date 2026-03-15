from huggingface_hub import login
import os
import json
import time
import asyncio

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bertscore

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bleurt_pytorch import BleurtTokenizer,BleurtForSequenceClassification

from ragas.metrics.collections import AnswerCorrectness
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory

from groq import Groq
from openai import AsyncOpenAI

def zeroshot_answer(question):

    prompt = f"""Você é um assistente educacional especializado na BNCC.
        Responda em no máximo 3 frases curtas.
        Não inclua explicações adicionais.

        Pergunta:
        {question}

        Resposta:
        """

    response = groq_client.chat.completions.create(
        model=modelname,
        messages=[
            {
                "role": "system",
                "content": "Você é um assistente confiável que responde em português de forma clara e objetiva."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
        max_tokens=150
    )

    return response.choices[0].message.content.strip()

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

for modelname in ["llama-3.3-70b-versatile","llama-3.1-8b-instant"]:
    modeloNome = "zeroshot_"+modelname

    # RAgas with problems - <exception>    The output is incomplete due to a max_tokens length limit.</exception>
    #llm = llm_factory("gpt-4o-mini", client=openai_client,max_tokens=2048)
    #embeddings = embedding_factory("openai", model="text-embedding-3-small", client=openai_client)
    rows = []

    for file in os.listdir(faq_dir):

        if not file.endswith(".tsv"):
            continue

        path = os.path.join(faq_dir, file)
        df = pd.read_csv(path, sep="\t")

        for _, row in df.iterrows():

            question = str(row["question"])
            ground_truth = str(row["answer"])

            model_answer = zeroshot_answer(question)
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



"""**Erro bizarro no RAGAS com GPT-4**

IncompleteOutputException: The output is incomplete due to a max_tokens length limit.

The above exception was the direct cause of the following exception:

RetryError                                Traceback (most recent call last)
RetryError: RetryError[<Future at 0x7d88cfe13fe0 state=finished raised IncompleteOutputException>]

The above exception was the direct cause of the following exception:

InstructorRetryException                  Traceback (most recent call last)
/usr/local/lib/python3.12/dist-packages/instructor/core/retry.py in retry_async(func, response_model, args, kwargs, context, max_retries, strict, mode, hooks)
    440     except RetryError as e:
    441         logger.debug(f"Retry error: {e}")
--> 442         raise InstructorRetryException(
    443             e.last_attempt._exception,
    444             last_completion=response,

InstructorRetryException: <failed_attempts>

<generation number="1">
<exception>
    The output is incomplete due to a max_tokens length limit.
</exception>
<completion>
    ChatCompletion(id='chatcmpl-DJMB6uhCJxgUsrXyVwffS8n4OA4FU', choices=[Choice(finish_reason='length', index=0, logprobs=None, message=ChatCompletionMessage(content='{\n    "TP": [\n        {\n            "statement": "No quinto ano, as habilidades da BNCC computação a serem trabalhadas incluem identificar e descrever características básicas de computadores e dispositivos móveis.",\n            "reason": "This statement is supported by the ground truth which discusses the identification and description of objects in the digital realm, although it does not explicitly mention computers and mobile devices."\n        },\n        {\n            "statement": "No quinto ano, as habilidades da BNCC computação a serem trabalhadas incluem utilizar ferramentas de edição de texto e imagem para criar e compartilhar conteúdo digital.",\n            "reason": "This statement aligns with the ground truth that mentions the use of computational tools for creating various content, which includes text and images."\n        },\n        {\n            "statement": "No quinto ano, as habilidades da BNCC computação a serem trabalhadas incluem identificar e descrever características básicas de redes de computadores.",\n            "reason": "This statement is somewhat supported by the ground truth which discusses the recognition of digital objects, although it does not specifically mention computer networks."\n        }\n    ],\n    "FP": [],\n    "FN": [\n        {\n            "statement": "As ...
</completion>
</generation>

<generation number="2">
<exception>
    The output is incomplete due to a max_tokens length limit.
</exception>
<completion>
    ChatCompletion(id='chatcmpl-DJMBhC1aR8RpMgVoOj6qiZIqe19Go', choices=[Choice(finish_reason='length', index=0, logprobs=None, message=ChatCompletionMessage(content='{\n    "TP": [\n        {\n            "statement": "No quinto ano, as habilidades da BNCC computação a serem trabalhadas incluem identificar e descrever características básicas de computadores e dispositivos móveis.",\n            "reason": "This statement is supported by the ground truth which discusses the identification and description of objects in the digital realm, although it does not explicitly mention computers and mobile devices."\n        },\n        {\n            "statement": "No quinto ano, as habilidades da BNCC computação a serem trabalhadas incluem utilizar ferramentas de edição de texto e imagem para criar e compartilhar conteúdo digital.",\n            "reason": "This statement aligns with the ground truth that mentions the use of computational tools for creating various content, which includes text and images."\n        },\n        {\n            "statement": "No quinto ano, as habilidades da BNCC computação a serem trabalhadas incluem identificar e descrever características básicas de redes de computadores.",\n            "reason": "This statement is somewhat supported by the ground truth which includes the recognition of digital objects, although it does not specifically mention computer networks."\n        }\n    ],\n    "FP": [],\n    "FN": [\n        {\n            "statement": "As h...
</completion>
</generation>

<generation number="3">
<exception>
    The output is incomplete due to a max_tokens length limit.
</exception>
<completion>
    ChatCompletion(id='chatcmpl-DJMCExvK7gUPQS5A2Jt7sqk5V4PVV', choices=[Choice(finish_reason='length', index=0, logprobs=None, message=ChatCompletionMessage(content='{\n    "TP": [\n        {\n            "statement": "No quinto ano, as habilidades da BNCC computação a serem trabalhadas incluem identificar e descrever características básicas de computadores e dispositivos móveis.",\n            "reason": "This statement is supported by the ground truth which discusses the identification and description of objects in the digital world."\n        },\n        {\n            "statement": "No quinto ano, as habilidades da BNCC computação a serem trabalhadas incluem utilizar ferramentas de edição de texto e imagem para criar e compartilhar conteúdo digital.",\n            "reason": "This statement aligns with the ground truth that mentions the use of computational tools for creating various content."\n        },\n        {\n            "statement": "No quinto ano, as habilidades da BNCC computação a serem trabalhadas incluem identificar e descrever características básicas de redes de computadores.",\n            "reason": "This statement is supported by the ground truth which includes the recognition of digital objects and their characteristics."\n        }\n    ],\n    "FP": [],\n    "FN": [\n        {\n            "statement": "As habilidades da BNCC Computação previstas para serem trabalhadas no quinto ano envolvem o desenvolvimento do pensamento computacional.",\n          ...
</completion>
</generation>

</failed_attempts>

<last_exception>
    The output is incomplete due to a max_tokens length limit.
</last_exception>
"""
