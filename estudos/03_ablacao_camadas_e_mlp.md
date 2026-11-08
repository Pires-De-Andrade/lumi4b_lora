# 03. Ablação por camadas e por tipo de bloco

Pergunta: **qual parte do adaptador causa a perda na múltipla escolha, e qual carrega o raciocínio em português?** Os estudos de SFT ([02](02_sft_dados_e_professor.md)) mostraram a perda e não mostraram a causa. Este estudo abre o adaptador por camada e por tipo de bloco.

## A arquitetura

O Lumi tem 42 camadas numeradas de 0 a 41, e cada uma é **um** bloco, de um de três tipos, num padrão fixo (`M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-`, em que `M` é Mamba2, `-` é MLP e `*` é atenção):

| Tipo de bloco | Camadas | Módulos com LoRA no adaptador completo | Observação |
| --- | --- | --- | --- |
| Mamba2 | 21 | `in_proj` (21) | o `out_proj` fica fora (o kernel fundido o ignora) |
| MLP | 17 | `up_proj` e `down_proj` (34) | |
| Atenção | 4 (camadas 12, 17, 24 e 32) | `q/k/v/o_proj` (16) | |

Total: 71 módulos, 17,05 milhões de parâmetros. **Camadas 0 a 20** (primeira metade): 10 Mamba, 9 MLP, 2 atenção (36 módulos). **Camadas 21 a 41:** 11 Mamba, 8 MLP, 2 atenção (35 módulos).

## E9. Triagem pós-treino (sem treinar)

**Hipóteses:**
- **H9a:** a perda na múltipla escolha vem de algumas partes do adaptador, não do conjunto.
- **H9b:** o idioma do raciocínio é decidido pelas camadas finais.
- **H9c:** os MLPs concentram ganhos de capacidade, e o Mamba e a atenção concentram o dano.

**Desenho:** o adaptador `adalberto` (treinado com os 71 módulos juntos) é fundido ao Lumi **só em parte**, com o efeito dos demais módulos zerado (`scripts/merge_adapter_layers.py`, na CPU; 7 a 9 min de avaliação por variante, só com thinking). Cinco variantes: completo, camadas 0 a 20, camadas 21 a 41, sem MLP (Mamba e atenção) e só MLP.

| Parte mantida | Módulos | GSM8K | Múltipla escolha | Fecha `</think>` (MCQ) | Traço em português (MCQ) |
| --- | --- | --- | --- | --- | --- |
| Lumi original | 0 | 81,7% | 69,6% | 86,2% | 0% |
| adaptador completo | 71 | 83,3% (+1,6) | 62,3% ▼ (−7,3) | 86,8% | 99,4% |
| camadas 0 a 20 | 36 | 83,6% (+1,9) | 62,6% ▼ (−7,0) | 81,3% | **0%** |
| camadas 21 a 41 | 35 | 76,6% ▼ (−5,0) | **18,4%** ▼ (−51,2) | 22,9% | 98,9% |
| sem MLP | 37 | 81,7% (0,0) | 60,0% ▼ (−9,6) | 81,3% | 98,7% |
| **só MLP** | 34 | **83,9%** (+2,2) | **69,6%** (0,0) | 87,8% | **2,4%** |

(▼: o IC 95% da diferença contra o Lumi fica abaixo de 0. Intervalo da linha "só MLP" na múltipla escolha: −3,4 a +3,4.)

**Leituras:**
1. **Só os MLPs do adaptador** mantêm a múltipla escolha do Lumi e dão o ganho de matemática (+2,2; IC −0,3 a +4,7, **não** significativo sozinho), mas o raciocínio continua em inglês (2,4% em português).
2. **O dano na múltipla escolha vem do Mamba e da atenção**, e **não do idioma do traço**: nas camadas 0 a 20 o traço sai em inglês (0% em português) e a múltipla escolha perde os mesmos 7 pontos do adaptador completo. Isso enfraquece a explicação "o português em si causa a perda".
3. **O idioma do raciocínio parece ser decidido no fim da pilha**: só as camadas 21 a 41 produzem português (98,9%). Mas esse adaptador parcial **quebra o modelo** (fecha `</think>` em 22,9%), porque as camadas finais se ajustaram ao que as iniciais entregavam. Logo, a leitura "o fim carrega a tradução" é uma hipótese, não um resultado.

**Veredito provisório:** H9a apoiada; H9c apoiada **na triagem**, a verificar com treino; H9b apoiada em parte e confundida pela quebra do modelo.

**Ameaças à validade:** (i) zerar módulos depois de treinar tudo junto **não equivale** a treinar só neles; o resultado "só MLP" pode refletir um adaptador de baixo efeito, que quase não muda o comportamento do Lumi (o raciocínio nem muda de idioma); (ii) o ganho de +2,2 na matemática tem IC que inclui zero; (iii) uma execução por variante, 385 itens (±4 a 5 pontos); (iv) o tamanho do efeito (e do dano) pode depender da interação entre blocos.

## E10. SFT só com MLP, treinado do zero (`adalberto_mlp`)

**Pergunta:** um adaptador com LoRA apenas nos 34 módulos de MLP, treinado desde o início, aprende o raciocínio em português **e** mantém a múltipla escolha do Lumi?

**Desenho:** mesmos dados (`adalberto`, 14.386 exemplos) e mesmos hiperparâmetros (r=16, alpha=32, lr 1e-4, 1 época, 450 passos); só muda `--targets mlp`, que reduz o adaptador de 71 para 34 módulos (**8,5 milhões** de parâmetros, metade do completo). Implementação: `TARGETS` em `peft_setup.py`, `--targets` em `train_sft.py`, 4 testes em `tests/test_peft_targets.py`. Treino: 450 passos, **23,9 min**, ~6,1 mil tokens/s, pico de 18,3 GiB; perda 0,47 → 0,21 (a do adaptador completo: 0,44 → 0,21). Adaptador de 8.529.920 parâmetros em 34 módulos.

**Previsões** (para poder refutá-las):

| Resultado | O que indicaria |
| --- | --- |
| Traço em português ≥ 90% **e** múltipla escolha ≥ 66% | O MLP sozinho aprende o idioma sem o dano: o Mamba e a atenção não são necessários para o português, e a hipótese H9c ganha força |
| Traço em inglês, múltipla escolha ≈ Lumi | O MLP não carrega o idioma: o português exige o Mamba ou a atenção (com o dano) |
| Traço em português, múltipla escolha ≈ 62% | O dano não era só do Mamba e da atenção: vem do português ou dos dados |

**Estado: treinado, avaliação em andamento.** Resultados na linha `adalberto_mlp` de [resultados_auto.md](resultados_auto.md) (regerar com `scripts/gen_estudos_resultados.py`); preencher aqui a leitura quando sair.

## Próximo estudo proposto: grade 2 por 2

Para separar **tipo de bloco** (MLP contra Mamba e atenção) de **posição** (camadas 0 a 20 contra 21 a 41), quatro SFTs com os mesmos dados e hiperparâmetros, cada um com ~25 min de treino mais 12 min de avaliação: MLP nas camadas 0 a 20, MLP nas 21 a 41, Mamba e atenção nas 0 a 20, Mamba e atenção nas 21 a 41. Só vale a pena se o `adalberto_mlp` der um sinal claro. Requer estender `--targets` com uma faixa de camadas.
