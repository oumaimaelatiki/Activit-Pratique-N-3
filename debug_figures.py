import sys
sys.stdout.reconfigure(encoding="utf-8")
import json

records = json.load(open("output/chunks.json", encoding="utf-8"))
for r in records:
    if r["type"] == "text" and "Figure" in r["content"] and r["page"] in (3, 4, 9):
        print(r["chunk_id"], "page", r["page"])
        print(r["content"])
        print()