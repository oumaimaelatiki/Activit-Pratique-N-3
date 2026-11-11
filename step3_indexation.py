# step3_indexation.py
import hashlib
import json
from pathlib import Path

import chromadb

from step2_embeddings import embed_texts, embed_images

PDF_PATH = Path("data/attention is all you need.pdf")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "attention_multimodal"


# ---------- Hachage SHA-256 du PDF ----------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


# ---------- Indexation ----------
def index_document():
    doc_id = sha256_file(PDF_PATH)[:16]   # identifiant court du document
    print("doc_id (SHA-256) :", doc_id)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # similarité cosinus
    )

    # Gestion des réindexations : si ce document est déjà indexé, on s'arrête
    existing = collection.get(where={"doc_id": doc_id}, limit=1)
    if existing["ids"]:
        print("Document déjà indexé, rien à faire.")
        return collection

    records = json.loads(Path("output/chunks.json").read_text(encoding="utf-8"))
    text_recs = [r for r in records if r["type"] == "text"]
    img_recs = [r for r in records if r["type"] == "image"]

    # --- Textes ---
    text_vecs = embed_texts([r["content"] for r in text_recs])
    collection.add(
        ids=[f"{doc_id}_{r['chunk_id']}" for r in text_recs],
        embeddings=text_vecs.tolist(),
        documents=[r["content"] for r in text_recs],
        metadatas=[{
            "doc_id": doc_id,
            "type": "text",
            "page": r["page"],
            "image_path": "",
        } for r in text_recs],
    )

    # --- Images ---
    img_vecs = embed_images([r["image_path"] for r in img_recs])
    collection.add(
        ids=[f"{doc_id}_img_{Path(r['image_path']).stem}" for r in img_recs],
        embeddings=img_vecs.tolist(),
        documents=[f"Figure de la page {r['page']}" for r in img_recs],
        metadatas=[{
            "doc_id": doc_id,
            "type": "image",
            "page": r["page"],
            "image_path": r["image_path"],
        } for r in img_recs],
    )

    print(f"{len(text_recs)} textes et {len(img_recs)} images indexés.")
    return collection


if __name__ == "__main__":
    col = index_document()
    print("Total dans la base :", col.count())