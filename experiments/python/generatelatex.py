import pandas as pd
import numpy as np
import os
import math

def truncate(x, decimals=2):
    factor = 10 ** decimals
    return math.trunc(x * factor) / factor

csv_dir = "../results/llama3-8b"#"../results/llama3-70b"

metrics = ["bleu", "rougeL", "bleurt", "bertscore", "entailment_prob", "contradiction_prob"]

for file in os.listdir(csv_dir):

    if not file.endswith(".csv"):
        continue

    path = os.path.join(csv_dir, file)

    df = pd.read_csv(path,sep=',')

    latex_parts = []

    for m in metrics:

        mean = truncate(df[m].mean(), 2)
        std = truncate(df[m].std(), 2)

        latex = f"{m}: ${mean:.2f} \\pm {std:.2f}$"
        latex_parts.append(latex)

    latex_line = " & ".join(latex_parts) + " \\\\"

    print(file)
    print(latex_line)
    print()
