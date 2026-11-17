SEJM_TERM ?= 9
WORKERS   ?= 4
MAX_DAYS  ?=
TEXT      ?= Musimy obniżyć podatki dla przedsiębiorców i ograniczyć biurokrację.

.DEFAULT_GOAL := help
.PHONY: help setup fetch prepare train predict notebook clean clean-all all

help:
	@echo "Kolejność kroków:"
	@echo "  make setup                  1. Zainstaluj środowisko (uv sync)"
	@echo "  make fetch                  2. Pobierz posłów i wypowiedzi z API (np. WORKERS=16 MAX_DAYS=3)"
	@echo "  make prepare                3. Złącz, wyczyść, oznacz strony i podziel zbiór"
	@echo "  make train                  4. Wytrenuj model (HerBERT + Logistic Regression)"
	@echo ""
	@echo "Dodatkowe:"
	@echo "  make all                    Wykonaj wszystko po kolei (setup -> fetch -> prepare -> train)"
	@echo "  make predict TEXT=\"...\"     Sklasyfikuj podaną wypowiedź"
	@echo "  make notebook               Otwórz presentation.ipynb"
	@echo "  make clean                  Usuń wygenerowane pliki i modele (zostawia pobrane speeches.csv)"
	@echo "  make clean-all              Usuń wszystko łącznie z pobranymi danymi"

setup:
	uv sync

fetch:
	uv run fetch-representatives --term $(SEJM_TERM)
	uv run fetch-speeches --term $(SEJM_TERM) --workers $(WORKERS) $(if $(MAX_DAYS),--max-days $(MAX_DAYS),)

prepare:
	uv run merge-speeches --term $(SEJM_TERM)
	uv run clean-speeches
	uv run label-sides
	uv run split-dataset

train:
	uv run train-baseline

predict:
	uv run predict-side "$(TEXT)"

notebook:
	uv run jupyter notebook src/data_mining_project/notebooks/presentation.ipynb

clean:
	rm -rf models data/term$(SEJM_TERM)/splits data/term$(SEJM_TERM)/embeddings
	rm -f data/term$(SEJM_TERM)/dataset.csv data/term$(SEJM_TERM)/dataset_clean.csv data/term$(SEJM_TERM)/dataset_labeled.csv

clean-all:
	rm -rf data models

all: setup fetch prepare train
