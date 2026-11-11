# step4_retrieval.py
import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

import chromadb
from PIL import Image

from step2_embeddings import embed_texts, embed_images

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "attention_multimodal"

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(COLLECTION_NAME)


def is_image_path(query: str) -> bool:
    return Path(query).suffix.lower() in {".png", ".jpg", ".jpeg"} and Path(query).exists()


def search(query: str, k: int = 5, content_type: str | None = None, show: bool = False):
    """
    Recherche par similarité cosinus.
    - query : une question en texte OU le chemin d'une image
    - content_type : None (tout), "text" ou "image"
    - show : ouvre les images trouvées dans la visionneuse de Windows
    """
    if is_image_path(query):
        query_vec = embed_images([query])
        print(f"\n=== Requête IMAGE : {query} ===")
    else:
        query_vec = embed_texts([query])
        print(f"\n=== Requête TEXTE : {query} ===")

    where = {"type": content_type} if content_type else None
    res = collection.query(
        query_embeddings=query_vec.tolist(),
        n_results=k,
        where=where,
    )

    results = []
    for i in range(len(res["ids"][0])):
        meta = res["metadatas"][0][i]
        score = 1 - res["distances"][0][i]   # distance cosinus -> similarité
        item = {
            "score": score,
            "type": meta["type"],
            "page": meta["page"],
            "image_path": meta["image_path"],
            "content": res["documents"][0][i],
        }
        results.append(item)

        print(f"\n#{i + 1}  score={score:.3f}  [{item['type']}]  page {item['page']}")
        if item["type"] == "text":
            print("   ", item["content"])
        else:
            print("    image :", item["image_path"])
            if show:
                Image.open(item["image_path"]).show()
    return results


if __name__ == "__main__":
    # Test 1 : texte -> tout (textes et images mélangés)
    search("multi-head attention", k=5)

    # Test 2 : texte -> images seulement
    search("architecture of the Transformer encoder and decoder", k=3, content_type="image")

    # Test 3 : image -> textes les plus proches
    search("output/images/page4_img1.png", k=3, content_type="text")