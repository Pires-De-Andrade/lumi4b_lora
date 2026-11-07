"""Registro de GPU para A-14: processos do nvidia-smi no início e no fim, e pico de memória do PyTorch."""
import json
import subprocess
import time
from pathlib import Path

import torch

GIB = 1024**3


def _smi(query: str, fmt: str) -> list[str]:
    try:
        out = subprocess.run(["nvidia-smi", f"--query-{query}", f"--format=csv,{fmt}"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception as e:  # noqa: BLE001
        return [f"erro: {e}"]
    return [line.strip() for line in out.splitlines() if line.strip()]


def snapshot() -> dict:
    """Quem está na GPU agora. Outro usuário distorce tempo e memória, então todo run grava isto."""
    free, total = torch.cuda.mem_get_info()
    return {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "free_gib": round(free / GIB, 2), "total_gib": round(total / GIB, 2),
        "compute_apps": _smi("compute-apps=pid,process_name,used_memory", "noheader"),
        "gpu": _smi("gpu=utilization.gpu,memory.used,temperature.gpu,clocks.sm", "noheader"),
    }


def reset_peak() -> None:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def peak_gib() -> dict:
    torch.cuda.synchronize()
    return {"peak_allocated_gib": round(torch.cuda.max_memory_allocated() / GIB, 2),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / GIB, 2)}


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
