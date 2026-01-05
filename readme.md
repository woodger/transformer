# 🚀 Transformer Inference Service

Высокопроизводительный **ML inference-сервис** на базе:

* **PyTorch Transformer**
* **CUDA / CPU**
* **Apache Arrow**
* **CLI-управление (argparse)**

Проект предназначен для **низколатентного и высокопропускного инференса** с чётким RPC-контрактом.

---

## 📌 Основные возможности

* ⚡ GPU-ускоренный инференс (CUDA)
* 🔄 Zero-copy транспорт данных (Apache Arrow)
* 🧠 Гибко настраиваемый Transformer
* 🧩 Чёткое разделение слоёв (RPC / ML / Schema)
* 🧪 Простота тестирования и масштабирования
* 🧑‍💻 Управление через CLI

---

## 📦 Установка

### Требования

* Python ≥ 3.9
* CUDA (опционально)
* PyTorch
* PyArrow

```bash
pip install torch pyarrow pytest
```

---

## ▶ Запуск сервиса

```bash
python ./app/main.py \
  --device gpu \
  --mode train \
  --epochs 50 \
  --patience 5 \
  --lr 1e-3 \
  --amp \
  --model-path ./model_fp16.pth
```

---

## ⚙ CLI параметры

### 🔧 Устройство и пути

| Аргумент       | Описание                           | По умолчанию        |
| -------------- | ---------------------------------- | ------------------- |
| `--device`     | Принудительный выбор `cpu` / `gpu` | auto                |
| `--model-path` | Путь к весам модели                | `model_weights.pth` |
| `--preds-path` | Путь сохранения предсказаний       | `/tmp/preds.arrow`  |
| `--pred-col`   | Название колонки с предсказаниями  | `y1`                |

---

### 🧠 Параметры модели

| Аргумент    | Описание                   | Default |
| ----------- | -------------------------- | ------- |
| `--seq-len` | Длина последовательности   | 10      |
| `--hidden`  | Размер скрытого слоя       | 128     |
| `--layers`  | Количество слоёв           | 3       |
| `--nhead`   | Количество attention-голов | 4       |
| `--dropout` | Dropout                    | 0.1     |

---

### 🏋️ Training (задел на будущее)

| Аргумент         | Описание         |
| ---------------- | ---------------- |
| `--lr`           | Learning rate    |
| `--batch-size`   | Размер батча     |
| `--epochs`       | Количество эпох  |
| `--patience`     | Early stopping   |
| `--alpha`        | Регуляризация    |
| `--weight-decay` | L2 регуляризация |

> ⚠ В текущей версии используются только для совместимости CLI.

---

## 📄 Arrow Schema

### Входные данные (`schema.py`)

```python
src: List[int64]  # shape: [batch, seq]
tgt: List[int64]  # shape: [batch, seq]
```

### Выходные данные

```python
out: List[float32]  # shape: [batch, seq * vocab]
```


---

## 🧪 Тестирование

* Модель можно тестировать отдельно (`model.py`)
* Инференс — через `InferenceService`
* RPC слой — мокается через Arrow Table

Запуск тестов. Из корня проекта:

```sh
PYTHONPATH=./app pytest -v
```

---

## 🚀 Масштабирование и расширение

Проект легко расширяется:

* 🔹 Dynamic batching
* 🔹 Multi-GPU (NCCL)
* 🔹 AMP / FP16
* 🔹 Model hot-reload
* 🔹 Prometheus metrics
* 🔹 Kubernetes + autoscaling

---

## 🏁 Заключение

Этот проект:

* готов для **production inference**
* легко деплоится
* масштабируется горизонтально
* подходит для **real-time и batch inference**
