"""Copy markdown text from the percent-format source into an already-executed notebook, keeping code outputs.

Fails if the code cells differ (then the notebook must be rebuilt and re-executed).
Usage: python tools/sync_md.py notebooks/src/A1_vision_transformers.py notebooks/A1_vision_transformers.ipynb
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from py2nb import parse_cells, to_notebook  # noqa: E402

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
fresh = to_notebook(parse_cells(src.read_text(encoding="utf-8")))
done = json.loads(dst.read_text(encoding="utf-8"))
assert len(fresh["cells"]) == len(done["cells"]), f"número de células difere: {len(fresh['cells'])} vs {len(done['cells'])}"
changed = 0
for new, old in zip(fresh["cells"], done["cells"]):
    assert new["cell_type"] == old["cell_type"], "tipos de célula diferem"
    if new["cell_type"] == "code":
        assert "".join(new["source"]) == "".join(old["source"]), "código mudou — reexecute o notebook:\n" + "".join(new["source"])[:200]
    elif new["source"] != old["source"]:
        old["source"] = new["source"]; changed += 1
dst.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{dst.name}: {changed} células markdown atualizadas")
