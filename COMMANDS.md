# Commands — second term + party (club) classification

Use the project venv (`uv run ...`). Fetching term 10 can take a long time.

## 1. Setup

```bash
uv sync
```

## 2. Download term 10 (run yourself)

```bash
uv run fetch-representatives --term 10
uv run fetch-speeches --term 10 --workers 16
```

Optional smoke test (only first 3 sitting days):

```bash
uv run fetch-speeches --term 10 --workers 8 --max-days 3
```

Or via Make:

```bash
make fetch10 WORKERS=16
```

## 3. Prepare each term

```bash
# Term 10
uv run merge-speeches --term 10
uv run clean-speeches --input data/term10/dataset.csv --output data/term10/dataset_clean.csv
uv run label-sides --input data/term10/dataset_clean.csv --output data/term10/dataset_labeled.csv

# Term 9 (refresh labels / cleaning if needed)
uv run merge-speeches --term 9
uv run clean-speeches --input data/term9/dataset.csv --output data/term9/dataset_clean.csv
uv run label-sides --input data/term9/dataset_clean.csv --output data/term9/dataset_labeled.csv
```

Or:

```bash
make prepare-term SEJM_TERM=10
make prepare-term SEJM_TERM=9
```

## 4. Combine terms + split

```bash
uv run combine-datasets --terms 9,10 --output data/combined/dataset_labeled.csv
uv run split-dataset --input data/combined/dataset_labeled.csv --output-dir data/combined/splits
```

Or:

```bash
make prepare-combined
```

## 5. Train side + club (TF-IDF)

```bash
uv run train-baseline --splits-dir data/combined/splits --output-dir models
```

Produces:

- `models/side_tfidf.joblib`
- `models/club_tfidf.joblib`
- `models/baseline_metrics.json`

## 6. Predict

```bash
uv run predict-side "Musimy podnieść płacę minimalną i chronić prawa pracownicze."
uv run predict-club "Musimy podnieść płacę minimalną i chronić prawa pracownicze."
```

## 7. Presentation notebook

```bash
uv run jupyter notebook src/data_mining_project/notebooks/presentation.ipynb
```

Select kernel **Python (data-mining-project)** / `.venv`.
