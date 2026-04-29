# Research Gap Detection Agent

Sistema para identificação automática de lacunas de pesquisa a partir de artigos científicos, utilizando:

- Modelagem de tópicos (BERTopic)
- Embeddings semânticos
- Grafos de coocorrência
- Busca vetorial (FAISS)
- Avaliação com LLM

---

## Dataset

Utiliza o dataset do ArXiv:

👉 https://www.kaggle.com/datasets/Cornell-University/arxiv?resource=download

### Estrutura esperada

Coloque o arquivo em:

- data/arxiv.json

---

## ⚙️ Instalação

### 1. Criar ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

## Como executar

O sistema é dividido em duas etapas:

---

## 1. Build do índice (offline)

Processa os artigos e constrói:

- embeddings  
- índice FAISS  
- modelo de tópicos  
- grafo de pesquisa  
- lacunas candidatas  

```bash
python build_index.py
```

## Saídas geradas:
artifacts/
├── faiss.index
├── embeddings.npy
├── mapping.json
├── gaps.json
├── config.json
└── research_graph.gexf

---

## 2.Consulta e análise (online)

Avalia as lacunas com LLM e gera relatório:

```bash
python query_system.py
```

## 3. Visualização

Abra o grafo no Gephi:

artifacts/research_graph.gexf

## Como funciona

1. Modelagem de tópicos

Usa BERTopic para extrair tópicos e palavras-chave.

2. Construção de grafo

Nós = conceitos
Arestas = coocorrência
Pesos = frequência

3. Identificação de lacunas

Uma lacuna é definida como:

combinação de conceitos relevantes que não aparecem juntos, mas estão em regiões densas do grafo

4. Ranking de lacunas

Baseado em:

centralidade no grafo
frequência dos conceitos

5. Validação

Busca vetorial (FAISS) para verificar ausência semântica.

6. Avaliação

LLM analisa:

viabilidade
impacto
risco