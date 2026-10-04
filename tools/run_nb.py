"""Execute a notebook cell by cell, logging progress and saving after every cell.

Usage: python tools/run_nb.py notebooks/A3_cnn_kaggle.ipynb [kernel_name]
"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

path = Path(sys.argv[1])
kernel = sys.argv[2] if len(sys.argv) > 2 else "infnet-venv"
class LiveClient(NotebookClient):
    """Echo stream outputs (prints) as they arrive, so long cells can be monitored."""
    def output(self, outs, msg, display_id, cell_index):
        if msg.get("msg_type") == "stream":
            txt = msg["content"].get("text", "")
            if "%|" not in txt:
                print("    | " + txt.rstrip().replace("\n", "\n    | "), flush=True)
        return super().output(outs, msg, display_id, cell_index)


nb = nbformat.read(path, as_version=4)
client = LiveClient(nb, timeout=None, kernel_name=kernel, resources={"metadata": {"path": str(path.parent)}})

t_start = time.time()
with client.setup_kernel():
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        t0 = time.time()
        first = cell.source.splitlines()[0][:70] if cell.source else ""
        print(f"[{i:3d}/{len(nb.cells)}] {first}", flush=True)
        try:
            client.execute_cell(cell, i)
        except Exception as e:
            nbformat.write(nb, path)
            print(f"ERRO na célula {i}: {type(e).__name__}: {str(e)[-3000:]}", flush=True)
            sys.exit(1)
        print(f"      ok em {time.time() - t0:.1f}s (total {time.time() - t_start:.0f}s)", flush=True)
        nbformat.write(nb, path)
print(f"FIM: {time.time() - t_start:.0f}s", flush=True)
