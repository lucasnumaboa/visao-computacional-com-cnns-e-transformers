"""Remove local machine paths from notebook outputs before publishing (cosmetic/privacy).

Usage: python tools/sanitize_outputs.py notebooks/*.ipynb
"""
import json
import re
import sys

PATTERNS = [
    (re.compile(r"[A-Za-z]:\\\\Users\\\\[^\\\\\"]+\\\\Downloads\\\\INFNET"), "."),          # JSON-escaped C:\\Users\\x\\Downloads\\INFNET
    (re.compile(r"[A-Za-z]:\\\\Users\\\\[^\\\\\"/]+"), "~"),                                 # JSON-escaped C:\\Users\\x
    (re.compile(r"[A-Za-z]:/Users/[^/\"]+"), "~"),                                           # C:/Users/x
]

for path in sys.argv[1:]:
    raw = open(path, encoding="utf-8").read()
    nb = json.loads(raw)
    changed = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        txt = json.dumps(cell.get("outputs", []), ensure_ascii=False)
        new = txt
        for pat, rep in PATTERNS:
            new = pat.sub(rep, new)
        if new != txt:
            cell["outputs"] = json.loads(new)
            changed += 1
    json.dump(nb, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{path}: {changed} células com caminhos locais limpos")
