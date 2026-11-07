"""Execute the optional learning notebook with this interpreter and a local kernel.

A temporary kernelspec pins sys.executable, avoiding accidental execution in a
global Python environment. Jupyter runtime/config files stay under the new output
directory. The source notebook remains output-free; executed output is separate.
"""

import argparse
import json
import os
import sys
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from app.evaluation.provenance import PROJECT_ROOT
from app.services.analysis.tracing import content_hash


@contextmanager
def local_kernel_environment(directory: Path):
    """Temporarily direct Jupyter/IPython state into this run, then restore it."""
    values = {
        "JUPYTER_PATH": str(directory),
        "JUPYTER_RUNTIME_DIR": str(directory / "runtime"),
        "JUPYTER_CONFIG_DIR": str(directory / "config"),
        "IPYTHONDIR": str(directory / "ipython"),
    }
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def execute_notebook(output_dir: Path) -> dict:
    # Optional imports keep normal backend/CI dependencies lightweight.
    import nbformat
    import torch
    from nbclient import NotebookClient

    source = PROJECT_ROOT / "notebooks/retrieval_with_pytorch.ipynb"
    source_bytes = source.read_bytes()
    notebook = nbformat.reads(source_bytes.decode("utf-8"), as_version=4)
    nbformat.validate(notebook)
    output_dir.mkdir(parents=True, exist_ok=False)
    with TemporaryDirectory(prefix="kernel-", dir=output_dir) as temporary:
        kernel_root = Path(temporary).resolve()
        if not kernel_root.is_relative_to(output_dir.resolve()):
            raise ValueError("Temporary kernel must stay inside the output directory")
        spec = kernel_root / "kernels/sacvoc-learning"
        spec.mkdir(parents=True)
        (spec / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        sys.executable,
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    "display_name": "SacVoc CPU retrieval",
                    "language": "python",
                }
            ),
            encoding="utf-8",
        )
        with local_kernel_environment(kernel_root):
            client = NotebookClient(
                notebook,
                timeout=120,
                kernel_name="sacvoc-learning",
                resources={"metadata": {"path": str(PROJECT_ROOT)}},
            )
            try:
                client.execute()
            finally:
                # Preserve useful cell output even if execution failed.
                nbformat.write(notebook, output_dir / "executed.ipynb")
    summary = {
        "torch_version": torch.__version__,
        "runtime_versions": {
            package: version(package)
            for package in ("numpy", "pydantic", "ipykernel", "nbclient")
        },
        "notebook_source_sha256": content_hash(source_bytes),
        "python_executable": sys.executable,
        "code_cells_executed": sum(cell.cell_type == "code" for cell in notebook.cells),
        "synthetic": True,
        "model_weights_loaded": False,
        "interpretation": "CPU retrieval math and NumPy parity checks; no training or quality claim.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = execute_notebook(args.output_dir)
    print(
        f"Executed {summary['code_cells_executed']} cells: {args.output_dir / 'executed.ipynb'}"
    )


if __name__ == "__main__":
    main()
