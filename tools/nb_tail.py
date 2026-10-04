"""Print the text outputs of the last executed code cells of a notebook (progress check)."""
import json
import sys

nb = json.load(open(sys.argv[1], encoding="utf-8"))
n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
done = [(i, c) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and c.get("outputs")]
total = sum(c["cell_type"] == "code" for c in nb["cells"])
print(f"células de código com output: {len(done)}/{total}")
for i, c in done[-n:]:
    print(f"--- célula {i}: {''.join(c['source'])[:80]!r}")
    for o in c["outputs"]:
        if o.get("output_type") == "stream":
            lines = "".join(o["text"]).splitlines()
            print("\n".join(l for l in lines if "%|" not in l)[-2500:])
        elif o.get("output_type") == "execute_result":
            print("".join(o["data"].get("text/plain", ""))[-2500:])
        elif o.get("output_type") == "error":
            print("ERRO:", o.get("ename"), o.get("evalue"))
