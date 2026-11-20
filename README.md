# data-mining-project

Predict political **side** (`left` / `center` / `right`) and parliamentary **club** from Polish Sejm speech transcripts.

## Quick start (term 9 + term 10)

Full command list: [`COMMANDS.md`](COMMANDS.md)

```bash
uv sync
make fetch10 WORKERS=16
make prepare-term SEJM_TERM=10
make prepare-term SEJM_TERM=9
make prepare-combined
make train
make notebook
```

Presentation notebook: `src/data_mining_project/notebooks/presentation.ipynb`
