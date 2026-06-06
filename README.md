# MLOps A3 — End-to-End MLOps Pipeline

**IIT Jodhpur | PGD AI Program | MLOps Group Assignment**

Binary sentiment classification on IMDb reviews using DistilBERT,
containerised with Docker, trained on Kaggle, tracked via W&B, and
automated through GitHub Actions.

---

## Links

| Resource | URL |
|---|---|
| Kaggle Notebook v1 | _paste link here_ |
| Kaggle Notebook v2 | _paste link here_ |
| Hugging Face Model | https://huggingface.co/pp2711/imdb-distilbert-sentiment |
| Docker Image | _paste link here_ |
| W&B Dashboard | _paste link here_ |

---

## Repository Structure

```
mlops-a3/
├── data/
│   └── prepare_data.py      # Task 2: data cleaning & normalisation
├── src/
│   ├── train.py             # Task 4 & 5: Kaggle training + HF push
│   └── inference.py         # Task 7: inference called by GitHub Actions
├── Dockerfile               # Task 6
├── requirements.txt
├── id2label.json            # label mapping (committed)
├── .github/
│   └── workflows/
│       ├── ci.yml           # Task 7.1: lint on push to develop
│       └── inference.yml    # Task 7.2: manual inference trigger
└── README.md
```

---

## Setup & Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare data
```bash
python data/prepare_data.py
```
Outputs `data/train.csv`, `data/test.csv`, and `id2label.json`.
> **Note:** Do not commit `train.csv` / `test.csv` — they are in `.gitignore`.

### 3. Train (Kaggle)
Upload `src/train.py`, `data/train.csv`, `data/test.csv`, and `id2label.json`
as a Kaggle dataset. Open a new Kaggle Notebook, enable GPU T4, add secrets
`WANDB_API_KEY` and `HF_TOKEN`, then run the notebook.

Change `VERSION = "v1"` → `"v2"` for the second experiment.

### 4. Docker — build & run locally
```bash
# Build
docker build --build-arg HF_MODEL_NAME=pp2711/imdb-distilbert-sentiment \
             -t mlops-a3-inference:latest .

# Run
docker run --rm \
  -e HF_TOKEN=<your_token> \
  -e INPUT_TEXT="This movie was absolutely fantastic!" \
  mlops-a3-inference:latest

# Push to Docker Hub
docker tag mlops-a3-inference:latest <your-dockerhub-username>/mlops-a3-inference:latest
docker push <your-dockerhub-username>/mlops-a3-inference:latest
```

### 5. GitHub Actions

**CI** — triggers automatically on every push to `develop`.

**Inference** — go to Actions → Inference → Run workflow → enter text → Run.

### 6. GitHub Secrets required
Add these in Settings → Secrets and Variables → Actions:
- `HF_TOKEN`
- `WANDB_API_KEY`

---

## Group Members & Contributions

| Name | Roll No | Contributions |
|---|---|---|
| Member 1 | — | Repo setup, GitHub Actions, Docker |
| Member 2 | — | Data preparation, model selection |
| Member 3 | — | Kaggle training, W&B tracking, HF push |
