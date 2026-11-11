# step5_tool.py

import sys
sys.stdout.reconfigure(encoding="utf-8")
from typing import Literal

from pathlib import Path

from langchain_core.tools import tool

from step4_retrieval import collection
from step2_embeddings import embed_texts, embed_images


def _is_image_path(query: str) -> bool:
    return Path(query).suffix.lower() in {".png", ".jpg", ".jpeg"} and Path(query).exists()


@tool
def retriever_tool(query: str, content_type: str = "all") -> str:
    """Recherche dans l'article scientifique "Attention Is All You Need" (Transformer).

    Utilise cet outil pour toute question sur le contenu de l'article : le texte
    (mécanisme d'attention, architecture, résultats, entraînement) ou les figures
    (schéma du Transformer, Scaled Dot-Product Attention, Multi-Head Attention).

    Arguments :
    - query : une requête courte, EN ANGLAIS (l'article est en anglais), par
      exemple "multi-head attention". Peut aussi être le chemin d'une image
      (.png, .jpg) pour chercher des passages similaires à cette image.
    - content_type : "text" pour chercher des passages de texte, "image" pour
      chercher des figures, "all" pour les deux. Par défaut "all".
    - k : nombre de résultats à retourner (par défaut 4).

    Retourne les meilleurs résultats avec leur type, leur page et leur score.
    Pour une figure, retourne le chemin du fichier image.
    """
    ct = content_type.lower()
    if "image" in ct and "text" in ct:
        content_type = "all"
    elif "image" in ct:
        content_type = "image"
    elif "text" in ct:
        content_type = "text"
    else:
        content_type = "all"
    k = 8  # au moins 4 résultats, au plus 8
    where = {"type": content_type} if content_type != "all" else None

    if _is_image_path(query):
        vec = embed_images([query])
    else:
        vec = embed_texts([query])

    res = collection.query(
        query_embeddings=vec.tolist(),
        n_results=k,
        where=where,
    )
    lines = []
    for i in range(len(res["ids"][0])):
        meta = res["metadatas"][0][i]
        score = 1 - res["distances"][0][i]
        if meta["type"] == "text":
            lines.append(
                f"[texte, page {meta['page']}, score {score:.2f}] {res['documents'][0][i]}"
            )
        else:
            lines.append(
                f"[image, PAGE {meta['page']}, score {score:.2f}] "
                f"fichier : {meta['image_path']}"
            )
    return "\n".join(lines) if lines else "Aucun résultat."


if __name__ == "__main__":
    print("Nom         :", retriever_tool.name)
    print("Description :", retriever_tool.description[:120], "...")
    print("Arguments   :", list(retriever_tool.args.keys()))
    print()
    print(retriever_tool.invoke({"query": "scaled dot-product attention"}))