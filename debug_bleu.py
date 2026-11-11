import sys
sys.stdout.reconfigure(encoding="utf-8")
import json

records = json.load(open("output/chunks.json", encoding="utf-8"))
for r in records:
    if r["type"] == "text" and "28.4" in r["content"]:
        print(r["chunk_id"], "page", r["page"])
        print(r["content"])
        print()