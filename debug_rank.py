import sys
sys.stdout.reconfigure(encoding="utf-8")

from step4_retrieval import collection
from step2_embeddings import embed_texts

query = "BLEU score of the big Transformer on English-to-German"
vec = embed_texts([query])
res = collection.query(query_embeddings=vec.tolist(), n_results=15,
                       where={"type": "text"})

for i, doc in enumerate(res["documents"][0], start=1):
    score = 1 - res["distances"][0][i - 1]
    mark = "  <== contient 28.4" if "28.4" in doc else ""
    print(f"#{i:2d} score={score:.2f} page {res['metadatas'][0][i-1]['page']}{mark}")