# viab:thinking contra base:nothinking, base:thinking (bootstrap pareado, 10000 reamostras, IC 95%, nota = strict_ok)

| Tarefa | n | viab:thinking | referência | referência (nota) | diferença | IC 95% | IC acima de 0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k_val | 1000 | 71.5% | base:nothinking | 81.4% | -9.9 | [-12.7, -7.1] | não |
| gsm8k_val | 1000 | 71.5% | base:thinking | 85.7% | -14.2 | [-17.1, -11.4] | não |
| mcq_val | 385 | 43.4% | base:nothinking | 63.9% | -20.5 | [-26.2, -15.1] | não |
| mcq_val | 385 | 43.4% | base:thinking | 69.6% | -26.2 | [-31.9, -20.5] | não |

## Forma da saída de viab:thinking

| Tarefa | fecha </think> e responde | traço em português (heurística) | truncadas | mediana de tokens |
| --- | --- | --- | --- | --- |
| gsm8k_val | 99.7% | pt 99.7% (indeterminado 0.2%) | 0.3% | 285 |
| mcq_val | 99.2% | pt 100.0% (indeterminado 0.0%) | 0.8% | 566 |

---- nota flexível ----
# viab:thinking contra base:nothinking, base:thinking (bootstrap pareado, 10000 reamostras, IC 95%, nota = flexible_ok)

| Tarefa | n | viab:thinking | referência | referência (nota) | diferença | IC 95% | IC acima de 0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k_val | 1000 | 80.3% | base:nothinking | 81.5% | -1.2 | [-3.4, +1.0] | não |
| gsm8k_val | 1000 | 80.3% | base:thinking | 85.7% | -5.4 | [-7.7, -3.1] | não |
| mcq_val | 385 | 53.0% | base:nothinking | 63.4% | -10.4 | [-15.6, -5.2] | não |
| mcq_val | 385 | 53.0% | base:thinking | 69.4% | -16.4 | [-21.8, -11.2] | não |


---- sem raciocínio: viab vs base ----
# viab:nothinking contra base:nothinking (bootstrap pareado, 10000 reamostras, IC 95%, nota = strict_ok)

| Tarefa | n | viab:nothinking | referência | referência (nota) | diferença | IC 95% | IC acima de 0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k_val | 1000 | 61.2% | base:nothinking | 81.4% | -20.2 | [-23.2, -17.2] | não |
| mcq_val | 385 | 43.1% | base:nothinking | 63.9% | -20.8 | [-26.2, -15.3] | não |
| Tarefa | n | viab:nothinking | referência | referência (nota) | diferença | IC 95% | IC acima de 0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k_val | 1000 | 77.6% | base:nothinking | 81.5% | -3.9 | [-6.1, -1.7] | não |
| mcq_val | 385 | 56.6% | base:nothinking | 63.4% | -6.8 | [-11.7, -2.1] | não |
