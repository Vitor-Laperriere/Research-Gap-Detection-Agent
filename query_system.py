import json
import numpy as np
import faiss
import tqdm

from sentence_transformers import SentenceTransformer
from openai import OpenAI


ARTIFACTS = "artifacts/"
MODEL = "deepseek-ai/deepseek-v4-pro"


client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="SUA-API-KEY-AQUI"
)


def llm(prompt):
    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1500
    )
    return res.choices[0].message.content


def load_artifacts():
    index = faiss.read_index(ARTIFACTS + "faiss.index")
    embeddings = np.load(ARTIFACTS + "embeddings.npy")

    with open(ARTIFACTS + "mapping.json") as f:
        mapping = json.load(f)

    return index, embeddings, mapping


def search(index, model, query, k=5):
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb), k)
    return I[0], D[0]


def identify_gaps(mapping):
    all_k = set()
    existing = set()

    for m in mapping:
        kws = set(m["keywords"])
        all_k.update(kws)

        for a in kws:
            for b in kws:
                if a != b:
                    existing.add(tuple(sorted((a, b))))

    gaps = []
    for a in all_k:
        for b in all_k:
            if a >= b:
                continue

            pair = tuple(sorted((a, b)))

            if pair not in existing and len(a) > 3 and len(b) > 3:
                gaps.append(pair)

    return gaps


def validate_gap(pair, index, model):
    query = " ".join(pair)
    _, dist = search(index, model, query, k=3)

    return float(np.mean(dist)) > 1.0


def evaluate_gaps(gaps):
    prompt = f"""
Avalie as lacunas:

{gaps[:20]}

Para cada:
- feasibility_score
- impact_score
- risk_score
- justificativa

JSON apenas.
"""
    return llm(prompt)


def main():
    index, embeddings, mapping = load_artifacts()

    model = SentenceTransformer("all-mpnet-base-v2")

    print("Identificando lacunas...")
    gaps = identify_gaps(mapping)
    print(f"Encontradas {len(gaps)} lacunas.")

    validated = []
    
    for g in tqdm.tqdm(gaps[:100], desc="Validando lacunas"):
        if validate_gap(g, index, model):
            validated.append(g)

    evaluation = evaluate_gaps(validated)

    report = llm(f"""
Gere relatório técnico:

LACUNAS:
{validated[:10]}

AVALIAÇÃO:
{evaluation}
""")

    print(report)


if __name__ == "__main__":
    main()