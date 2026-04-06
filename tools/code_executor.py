"""Sandboxed Python code execution via subprocess."""

from __future__ import annotations

import os
import subprocess
from typing import Any


def execute(code: str, output_dir: str, timeout: int = 30) -> dict[str, Any]:
    """Run *code* in a subprocess and return results.

    The code receives ``OUTPUT_DIR`` as a pre-set variable pointing to
    *output_dir* so it can save files (plots, data, etc.).

    Returns ``{"stdout", "stderr", "exit_code", "files"}``.
    """
    os.makedirs(output_dir, exist_ok=True)

    preamble = (
        "import os\n"
        f"OUTPUT_DIR = {output_dir!r}\n"
        "os.makedirs(OUTPUT_DIR, exist_ok=True)\n"
    )
    full_code = preamble + code

    env = {**os.environ, "MPLBACKEND": "Agg"}

    try:
        result = subprocess.run(
            ["python", "-c", full_code],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        exit_code = result.returncode
        stdout = result.stdout[:3000]
        stderr = result.stderr[:500]
    except subprocess.TimeoutExpired:
        exit_code = -1
        stdout = ""
        stderr = f"Code execution timed out after {timeout}s"

    files = sorted(os.listdir(output_dir)) if os.path.isdir(output_dir) else []

    return {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "files": files,
    }
