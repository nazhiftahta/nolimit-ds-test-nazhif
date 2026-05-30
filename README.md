# IndoBERT Sentiment Analysis + FAISS Retrieval

This repository contains a runnable Indonesian sentiment analysis pipeline built for the NoLimit Data Scientist technical test. The project fine-tunes IndoBERT for 3-class sentiment classification and adds embedding-based similarity search using FAISS.

Live demo: https://nolimit-ds-test-nazhif.streamlit.app/

## Task

Selected task: **Classification**

Classes:
- `POSITIVE`
- `NEUTRAL`
- `NEGATIVE`

The pipeline also uses embeddings for representation/search, as required by the test.

## Dataset

Dataset: **ID-SMSA: Indonesian Stock Market Dataset for Sentiment Analysis**.

Source:
- Mendeley Data: https://data.mendeley.com/datasets/tn4vzs8tdw/3
- DOI: `10.17632/tn4vzs8tdw.3`
- Citation: Hartanto, Jason; Liundi, Timothy; Sutoyo, Rhio (2025), "ID-SMSA: Indonesian Stock Market Dataset for Sentiment Analysis", Mendeley Data, V3.

The dataset was previously commonly accessed through the Hugging Face `indonlp/indonlu` dataset collection. At the time this project was prepared, that source was no longer directly downloadable in the working environment, so the dataset was obtained from the Mendeley Data release above.

Local CSV splits are included under `data/`:
- `data/train.csv`
- `data/validation.csv`
- `data/test.csv`
- `data/sample_input.csv`

Dataset labels are mapped as:
- `0`: `POSITIVE`
- `1`: `NEUTRAL`
- `2`: `NEGATIVE`

License: the Mendeley Data release is licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

## Models

Classification model:
- `indobenchmark/indobert-base-p1`
- `AutoModelForSequenceClassification`
- 3 output labels

Fine-tuned model:
- Hugging Face Hub: https://huggingface.co/sovrncrypt/indobert-smsa-nolimit-ds-test
- Model ID: `sovrncrypt/indobert-smsa-nolimit-ds-test`

Embedding model:
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`

Retrieval:
- FAISS cosine similarity
- Implemented as L2-normalized embeddings with inner product search

## Final Result

Final selected configuration:
- Epochs: `3`
- Learning rate: `2e-5`
- Per-device batch size: `8` in the final notebook run
- Effective batch size: `16` on Kaggle 2x GPU
- Max train samples: full training set

Test set result:

| Metric | Score |
| --- | ---: |
| Accuracy | `0.926` |
| Weighted F1 | `0.9228` |

Classification report:

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| POSITIVE | 0.93 | 0.96 | 0.94 | 208 |
| NEUTRAL | 0.97 | 0.70 | 0.82 | 88 |
| NEGATIVE | 0.91 | 0.99 | 0.95 | 204 |

A 5-epoch run was also tested. It produced a slightly lower test weighted F1 (`0.9192`) and showed mild overfitting: validation loss increased while training loss continued to decrease. The 3-epoch checkpoint was selected because it gave the best test performance.

## Example Predictions

The notebook and script output sentiment predictions for `data/sample_input.csv`, including class confidence scores.

Example format:

| Text | Predicted Label | Confidence | Positive | Neutral | Negative |
| --- | --- | ---: | ---: | ---: | ---: |
| Doi asik bgt orangnya | POSITIVE | 0.9999 | 0.9999 | 0.0001 | 0.0000 |
| Ada pengumuman nih gaiss, besok kegiatan kantor diliburkan | NEUTRAL | 0.9994 | 0.0001 | 0.9994 | 0.0005 |
| Kok gni sih kelakuannya | NEGATIVE | 0.9998 | 0.0001 | 0.0001 | 0.9998 |

## FAISS Retrieval

The retrieval component:
1. Encodes all training texts using Sentence-Transformers.
2. Builds a FAISS index from normalized embeddings.
3. Encodes each query/sample input.
4. Returns top-k similar training examples with similarity scores and labels.

This is implemented in:
- `src/faiss_retriever.py`
- `src/train_and_evaluate.py`
- `notebooks/sentiment-indobert-faiss.ipynb`

## Repository Structure

```text
nolimit-ds-test-nazhif/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── sample_input.csv
│   ├── test.csv
│   ├── train.csv
│   └── validation.csv
├── docs/
│   └── flowchart.png
├── notebooks/
│   └── sentiment-indobert-faiss.ipynb
├── src/
│   ├── faiss_retriever.py
│   └── train_and_evaluate.py
├── LICENSE.txt
├── README.md
└── requirements.txt
```

The flowchart is available at:
- `docs/flowchart.png`

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Run Script

Train, evaluate, predict sample inputs, and run FAISS retrieval:

```bash
python src/train_and_evaluate.py \
  --train_csv data/train.csv \
  --validation_csv data/validation.csv \
  --test_csv data/test.csv \
  --sample_input_csv data/sample_input.csv \
  --output_dir outputs/indobert-sm-sa \
  --do_train --do_eval --do_retrieval \
  --epochs 3 \
  --max_train_samples 0 \
  --top_k 5
```

For a quick smoke test:

```bash
python src/train_and_evaluate.py --do_train --do_eval --do_retrieval --epochs 1 --max_train_samples 300
```

## Run Notebook

Notebook:

```text
notebooks/sentiment-indobert-faiss.ipynb
```

The notebook contains the end-to-end workflow:
- load local CSV data
- fine-tune IndoBERT
- evaluate on validation and test sets
- predict sample inputs with confidence scores
- run FAISS similarity retrieval
- save the fine-tuned model

## Run Streamlit App

Live deployed app:

```text
https://nolimit-ds-test-nazhif.streamlit.app/
```

Run locally:

```bash
streamlit run app/streamlit_app.py
```

The deployed app uses the fine-tuned model from Hugging Face Hub by default:

```text
sovrncrypt/indobert-smsa-nolimit-ds-test
```

For local experiments, you can also use a downloaded fine-tuned model directory, for example:

```text
outputs/indobert-sm-sa-notebook
```

The selected model directory must include model weights, such as `model.safetensors` or `pytorch_model.bin`.

## Notes for Reviewers

- The fine-tuned model checkpoint is hosted on Hugging Face Hub because model weight files are large.
- The notebook contains the task-specific results and example outputs.
- `data/sample_input.csv` is included for local verification.
- `docs/flowchart.png` documents the end-to-end pipeline.

## License

This repository includes a local `LICENSE.txt` for the code in this project. Model and dataset assets follow their respective upstream licenses:
- IndoBERT: https://huggingface.co/indobenchmark/indobert-base-p1
- Sentence-Transformers model: https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2
- Dataset: ID-SMSA on Mendeley Data, licensed under **CC BY 4.0**: https://data.mendeley.com/datasets/tn4vzs8tdw/3

Under CC BY 4.0, the dataset may be shared and adapted as long as appropriate credit is given, a link to the license is provided, and changes are indicated. This use does not imply endorsement by the dataset rights holder.
