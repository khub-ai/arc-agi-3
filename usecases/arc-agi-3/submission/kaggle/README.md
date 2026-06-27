# Cognitive OS — ARC-AGI-3 Kaggle starter

The submission scaffold, aligned to the official `arcprize/ARC-AGI-3-Kaggle-Starter`.
It builds and pushes the Kaggle notebook that runs the Cognitive OS against the
competition's gateway sidecar. See `../KAGGLE_SUBMISSION.md` for the full mechanism.

The agent is **`agent/my_agent.py`** = `MyAgent(CosAgent)` — the only file with our
logic. Everything else mirrors the official starter.

## One-time

1. `cos-code` dataset: build the slice (`cd ..; python build_package.py` →
   `dist/cos-code`) and upload it as a private Kaggle Dataset named `cos-code`.
2. Model: attach **Qwen3-VL-8B-Instruct** (`qwen-lm/qwen-3-vl`,
   `transformers/8b-instruct` variation). It fits the 2× T4 in fp16 — no quantization.
3. Edit `notebooks/kernel-metadata.json`: replace `REPLACE_WITH_YOUR_USERNAME`
   (in `id` and the `cos-code` dataset source) and `REPLACE_WITH_QWEN3_VL_8B_MODEL_SLUG`.
4. Kaggle CLI token at `.kaggle/access_token` (one line, from kaggle.com/settings).

## Build + submit

```
make submit      # build_notebook.py -> notebooks/submission.ipynb, then kaggle kernels push
make status      # watch the most recent run
```

Then on kaggle.com: open the kernel, select `submission.parquet` from the output,
and click **Submit to Competition**. 5 submissions/day, 12-hour runtime.

## Local testing

The starter's `make play-local` uses its own venv, which does **not** have the COS
engine deps. For a real local COS test, use the engine environment instead:

```
cd ..                                  # usecases/arc-agi-3/submission
python run_framework_offline.py ls20   # drives CosAgent through Agent.main(), offline
```

## Accelerator

Set `ACCELERATOR` at the top of `scripts/build_notebook.py` (`t4` = T4×2, the
default; `rtx6000` = 24 GB single; `p100`; `cpu`). The serve cell loads the model
with transformers (`device_map="auto"` across whatever GPUs are present).
