import json
import itertools
import numpy as np
import faiss
import networkx as nx
import os

from collections import Counter

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


DATA_PATH = "data/arxiv.json"
ARTIFACTS = "artifacts/"
LIMIT = 20000


def load_data(path, limit):
    papers = []

    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break

            data = json.loads(line)

            if not data.get("abstract") or not data.get("title"):
                continue

            papers.append({
                "title": data["title"],
                "abstract": data["abstract"],
                "categories": data.get("categories", "")
            })

    return papers


def build_corpus(papers):
    return [f"{p['title']}. {p['abstract']}" for p in papers]


def build_embeddings(docs, model):
    emb = model.encode(docs, batch_size=32, show_progress_bar=True)
    return np.array(emb)


def build_faiss(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def build_topics(docs, model):
    topic_model = BERTopic(embedding_model=model, verbose=True)
    topics, _ = topic_model.fit_transform(docs)
    return topic_model, topics


def build_mapping(topic_model):
    mapping = []

    for topic_id in topic_model.get_topic_info()["Topic"]:
        if topic_id == -1:
            continue

        words = topic_model.get_topic(topic_id)

        mapping.append({
            "topic_id": int(topic_id),
            "keywords": [w[0] for w in words[:10]]
        })

    return mapping


# ===== NOVO: FILTRO DE KEYWORDS =====
def filter_keywords(mapping, min_freq=5):
    counter = Counter()

    for m in mapping:
        counter.update(m["keywords"])

    valid = {k for k, v in counter.items() if v >= min_freq and len(k) > 3}

    return valid, counter


# ===== NOVO: GRAFO COM PESOS =====
def build_graph(mapping, valid_keywords):
    G = nx.Graph()

    for m in mapping:
        kws = [k for k in m["keywords"] if k in valid_keywords]

        for a, b in itertools.combinations(kws, 2):
            if G.has_edge(a, b):
                G[a][b]["weight"] += 1
            else:
                G.add_edge(a, b, weight=1)

    nx.write_gexf(G, ARTIFACTS + "research_graph.gexf")
    return G


# ===== NOVO: LACUNAS INTELIGENTES =====
def compute_gaps(G, keyword_freq, top_k=200):
    centrality = nx.degree_centrality(G)

    nodes = list(G.nodes)

    gaps = []

    for a, b in itertools.combinations(nodes, 2):
        if G.has_edge(a, b):
            continue

        # só conceitos relevantes
        if centrality[a] < 0.05 or centrality[b] < 0.05:
            continue

        # score = centralidade - frequência (evita trivialidade)
        score = (
            centrality[a] + centrality[b]
        ) / (keyword_freq[a] + keyword_freq[b])

        gaps.append({
            "pair": (a, b),
            "score": float(score),
            "freq_a": int(keyword_freq[a]),
            "freq_b": int(keyword_freq[b])
        })

    gaps = sorted(gaps, key=lambda x: x["score"], reverse=True)

    return gaps[:top_k]


def save_artifacts(index, embeddings, mapping, gaps):
    faiss.write_index(index, ARTIFACTS + "faiss.index")
    np.save(ARTIFACTS + "embeddings.npy", embeddings)

    with open(ARTIFACTS + "mapping.json", "w") as f:
        json.dump(mapping, f)

    with open(ARTIFACTS + "gaps.json", "w") as f:
        json.dump(gaps, f, indent=2)

    with open(ARTIFACTS + "config.json", "w") as f:
        json.dump({"model": "all-mpnet-base-v2"}, f)


def main():
    os.makedirs(ARTIFACTS, exist_ok=True)

    papers = load_data(DATA_PATH, LIMIT)
    print(f"Carregados {len(papers)} papers.")

    docs = build_corpus(papers)

    model = SentenceTransformer("all-mpnet-base-v2")

    print("Construindo embeddings...")
    embeddings = build_embeddings(docs, model)

    print("Construindo índice FAISS...")
    index = build_faiss(embeddings)

    print("Construindo tópicos...")
    topic_model, topics = build_topics(docs, model)

    mapping = build_mapping(topic_model)

    print("Filtrando keywords...")
    valid_keywords, keyword_freq = filter_keywords(mapping)

    print(f"{len(valid_keywords)} keywords relevantes.")

    print("Construindo grafo...")
    G = build_graph(mapping, valid_keywords)

    print("Computando lacunas...")
    gaps = compute_gaps(G, keyword_freq)

    print(f"{len(gaps)} lacunas relevantes geradas.")

    save_artifacts(index, embeddings, mapping, gaps)

    print("Index build completo.")


if __name__ == "__main__":
    main()