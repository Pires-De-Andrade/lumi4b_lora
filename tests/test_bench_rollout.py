"""Benchmark de velocidade da geração do RL (scripts/bench_rollout.py): só as partes sem GPU."""
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("bench_rollout", Path(__file__).resolve().parents[1] / "scripts" / "bench_rollout.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

LOG = """INFO startup
(APIServer) INFO Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%
INFO Engine 000: Avg prompt throughput: 900.0 tokens/s, Avg generation throughput: 2000.0 tokens/s, Running: 128 reqs, Waiting: 40 reqs, GPU KV cache usage: 98.0%, Prefix cache hit rate: 10%
INFO Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 3000.0 tokens/s, Running: 100 reqs, Waiting: 0 reqs, GPU KV cache usage: 80.5%
INFO Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 1000.0 tokens/s, Running: 4 reqs, Waiting: 0 reqs, GPU KV cache usage: 5.0%
"""


def test_backend_padrao_nao_passa_mamba_config_e_liga_o_log_de_estatisticas():
    kw = B.llm_kwargs("m", 0.88, 128, True, "triton", 7)
    assert "mamba_config" not in kw and kw["enable_prefix_caching"] is True and kw["disable_log_stats"] is False
    assert kw["max_model_len"] == 5120 and kw["gpu_memory_utilization"] == 0.88 and kw["max_num_seqs"] == 128 and kw["seed"] == 7
    assert kw["generation_config"] == "vllm" and kw["dtype"] == "bfloat16"


def test_flashinfer_e_prefix_caching_desligado():
    kw = B.llm_kwargs("m", 0.93, 64, False, "flashinfer", 0)
    assert kw["mamba_config"] == {"backend": "flashinfer"} and kw["enable_prefix_caching"] is False and kw["gpu_memory_utilization"] == 0.93


def test_estatisticas_do_log_ignoram_linhas_sem_resposta_rodando():
    s = B.parse_log_stats(LOG, 128)
    assert s["linhas"] == 3 and s["rodando_max"] == 128 and s["rodando_mediana"] == 100
    assert s["fracao_no_maximo"] == round(1 / 3, 3) and s["esperando_medio"] == round(40 / 3, 1)
    assert s["kv_max_pct"] == 98.0 and s["kv_medio_pct"] == round((98.0 + 80.5 + 5.0) / 3, 1) and s["tok_s_medio_do_log"] == 2000


def test_log_sem_estatisticas():
    assert B.parse_log_stats("INFO nada\nINFO nada", 128) == {"linhas": 0}


def test_tabela_junta_execucao_e_estatisticas_pelo_rotulo(tmp_path):
    p = tmp_path / "r.jsonl"
    run = {"label": "base", "gen_seconds": 100, "gen_tokens": 300000, "tok_per_s": 3000, "truncated": 0.2, "reward_mean": {"mcq": 0.4, "math": 0.9}}
    p.write_text(json.dumps(run) + "\n" + json.dumps({"label": "base", "vllm_stats": B.parse_log_stats(LOG, 128)}) + "\n"
                 + json.dumps({"label": "sem_log", "gen_seconds": 90, "gen_tokens": 1, "tok_per_s": 1, "truncated": 0.0, "reward_mean": {}}) + "\n")
    t = B.report(p).splitlines()
    assert len(t) == 4 and t[2].startswith("| base | 100 | 300000 | 3000 | 0.200 | 0.4/0.9 | 77.3/128 |")
    assert "| sem_log | 90 |" in t[3] and t[3].endswith("| -/- | - | -/- |")
