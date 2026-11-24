# Free compute for the HFM projects

This folder is a small job runner on free, public compute. It exists because the
research container has 4 CPUs, no GPU and a restricted network.

## In use: GitHub Actions on this public repository

- **What it gives:** 4-core, 16 GB Linux machines, up to about 6 hours per job, many
  jobs in parallel, open internet access. Free for public repositories. No GPU.
- **How to run a job:** add or change a shell script in `jobs/`, then push to this
  branch. The workflow `.github/workflows/hfm-compute.yml` runs each changed script
  and commits its log (and any files written to `$OUT`) under `results/<job>/`.
- **What a job gets:** `$OUT` for results, `$DATA` for downloads, and
  `$GH_MODELS_TOKEN` for GitHub Models (see below).
- **Etiquette:** jobs download only public, openly licensed data and send a generic
  client name where an API asks for one.

Jobs run so far are in `jobs/`, with their logs in `results/`.

## Free options that need your own account

None of these can be created or used on your behalf without you signing up.

| Option | What it gives | Can it be automated? | What is needed |
| --- | --- | --- | --- |
| Kaggle Notebooks | GPU (T4 x2 or P100), about 30 hours a week, 12-hour sessions | Yes, through the Kaggle API (`kaggle kernels push`) | A phone-verified Kaggle account and its API token |
| Google Colab (free tier) | One T4 GPU, sessions that disconnect when idle | No, browser only | A Google account |
| Amazon SageMaker Studio Lab | One T4 GPU, 4-hour sessions | No, browser only | An approved Studio Lab account |
| Lightning AI | Monthly free GPU credits | Partly, through its CLI | An account |
| GitHub Models | Rate-limited calls to hosted language models | Yes, from the workflow here | Nothing extra if the probe in `jobs/models-probe.sh` succeeds |
| Research allocations (NSF ACCESS, EuroHPC, national centres) | Large GPU and CPU allocations | Yes, through batch schedulers | An application; days to weeks |

Kaggle is the practical route to a free GPU for the blocked projects (weather,
pathology, crystals, cell atlas, model training), because it is the only free GPU
with an API. With a Kaggle token stored as a repository secret, the same job
scripts can push a notebook to Kaggle and collect its output.

## What each kind of compute unblocks

- **GitHub Actions (CPU):** every project whose experiment is CPU-bound or needs
  data this container cannot reach. It already unblocked the cold-battery project
  by finding a working download for the Mendeley battery datasets.
- **GitHub Models:** small-scale tests for the six agent-harness projects.
- **A free GPU (Kaggle):** the 20 or so projects that need model inference or
  training. Budgets of 30 GPU-hours a week are enough for fine-tuning small models
  and for inference with the published science models, not for pretraining.
