# Código derivado do Antidoom (Liquid AI)

Origem: https://github.com/Liquid4All/antidoom, commit `bd6a126476e18554b0cacaea3fd9f258fdde1f97` (07/07/2026), licença Apache-2.0 (cópia em `LICENSE`).

Arquivos deste projeto com trechos derivados desse repositório:

- `src/lumi_ft/train/antidoom_oficial.py`: detector de repetição (`repetition.py`), escolha do token rejeitado com ajuste por tokens
  (`generate.py`), escolha dos tokens escolhidos (`sampling.py`), decodificação de tokens (`tokens.py`), regularização de frequência e
  amostragem (`ftpo_data.py`) e a conta da perda FTPO (`reference_ftpo_trainer.py`, `compute_loss`).
- `scripts/antidoom_oficial_train.py`: usa diretamente as classes do repositório oficial (clonado fora deste projeto) para o braço de conferência.

Mudanças em relação ao original:

- funções separadas do pipeline de geração (sem cliente HTTP, filas ou multiprocessamento) e adaptadas para receber ids e strings de token do tokenizer do Lumi;
- a perda é calculada por evento em FP32 (cabeça de saída em FP32), sem o `DPOTrainer`; a média por lote é feita fora da função;
- as palavras de recomeço de frase (`_RESTART_WORDS`) ganharam equivalentes em português, porque os traços são em português;
- a escolha dos tokens escolhidos usa os 20 maiores logits do modelo de partida no ponto do rejeitado (no original vêm das log-probabilidades
  devolvidas pelo vLLM durante a geração, do mesmo modelo).
