"""Treina um adaptador LoRA ou DoRA no Lumi (fases 2 a 4). Retomável: repetir o mesmo comando continua do último passo salvo.

Exemplos (sempre desacoplado: scripts/detach.sh --gpu <nome> ...):
  viabilidade:  PYTHONPATH=src python scripts/train_sft.py --arm lora --viab --epochs 1 --out runs/train/viab
  varredura:    PYTHONPATH=src python scripts/train_sft.py --arm lora --frac 0.2 --lr 5e-5 --out runs/train/lr5e-5
  mistura 100/0: ... --mix 100/0
  finalista:    PYTHONPATH=src python scripts/train_sft.py --arm dora --epochs 2 --out runs/train/dora_final
"""
import argparse
import json
import sys

from lumi_ft.train.sft import TrainConfig, train


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    d = TrainConfig()
    ap.add_argument("--arm", choices=["lora", "dora"], default=d.arm)
    ap.add_argument("--targets", choices=["all", "mlp"], default=d.targets, help="módulos do adaptador: os 71 do PRD ou só o MLP (34)")
    ap.add_argument("--r", type=int, default=d.r)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--eff-batch", type=int, default=d.eff_batch, help="sequências por passo do otimizador")
    ap.add_argument("--token-budget", type=int, default=0, help="tokens por micro-batch (0 = 2.048 no LoRA, 1.024 no DoRA)")
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--mix", choices=["75/25", "100/0"], default=d.mix)
    ap.add_argument("--frac", type=float, default=None, help="fração dos dados (0.2 = 20%%, por hash estável)")
    ap.add_argument("--viab", action="store_true", help="subconjunto de viabilidade (10.000 com raciocínio + 25%% de tokens sem)")
    ap.add_argument("--data", default=d.data)
    ap.add_argument("--valid", default=d.valid)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-every", type=int, default=d.save_every)
    ap.add_argument("--log-every", type=int, default=d.log_every)
    ap.add_argument("--max-steps", type=int, default=0, help="para depois de N passos (depuração)")
    a = ap.parse_args()
    cfg = TrainConfig(arm=a.arm, r=a.r, lr=a.lr, epochs=a.epochs, eff_batch=a.eff_batch, token_budget=a.token_budget,
                      seed=a.seed, mix=a.mix, frac=a.frac, viab=a.viab, data=a.data, valid=a.valid, out=a.out,
                      save_every=a.save_every, log_every=a.log_every, max_steps=a.max_steps, targets=a.targets)
    print(json.dumps(train(cfg), indent=1, default=str))
    print("EXIT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
