# Training Comparison Report

Дата: 2026-05-23

Метрика: чем ниже `ret_mae_skill`, тем лучше. Значения ниже `1.0` лучше baseline.

## Top 3

| Rank | Report | Scenario | seq_len | context | window | data span | hidden | layers | batch | best ret_mae_skill | improvement | best frame | best epoch | stage |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `models/plot-metrics.B8.jsonl` | вертикальное расширение истории | 10 | 10 | 60 | ~5 лет | 256 | 5 | 256 | 0.781150 | 21.89% | 6 | 25 | 4 |
| 2 | `models/plot-metrics.B5.jsonl` | расширение контекста | 23 | 23 | 60 | ~1 год | 256 | 5 | 256 | 0.817525 | 18.25% | 7 | 19 | 4 |
| 3 | `models/plot-metrics.jsonl` | расширение окна | 20 | 10 | 90 | ~1 год | 256 | 5 | 256 | 0.862324 | 13.77% | 3 | 3 | 1 |

## Baseline Control

| Report | Scenario | seq_len | context | window | data span | best ret_mae_skill | improvement | best frame | best epoch | stage |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `models/plot-metrics.B.jsonl` | базовый прогон за 1 год | 10 | 10 | 60 | ~1 год | 0.887687 | 11.23% | 4 | 14 | 4 |

## Notes

- Контрольная точка `seq_len=10, context=10, window=60, ~1 год` лежит в `models/plot-metrics.B.jsonl`; лучший `ret_mae_skill = 0.887687`.
- Вертикальное расширение истории с ~1 года до ~5 лет сейчас дает самый сильный прирост: `0.887687 -> 0.781150`.
- Расширение контекста до 23 инструментов на данных за ~1 год дает второй по силе результат: `0.817525`.
- Увеличение `window` с 60 до 90 не помогло при `seq_len=10`, но стало полезнее в связке с `seq_len=20`: `0.934526 -> 0.862324`.
- В последнем прогоне `window=90, seq_len=20` лучший checkpoint пришел на `loss_stage=1`; следующие этапы loss в этом прогоне не улучшили `ret_mae_skill`.
