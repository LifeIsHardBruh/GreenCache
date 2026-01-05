# GreenCache

GreenCache is a carbon-aware cache management framework that dynamically derives resource allocation plans for LLM serving. 

It includes:

- Dataset preprocessing and chat-history generation from ShareGPT dataset.
- Cache simulation tooling to build cache lists and request slices.
- A multi-round QA workload driver (LMCache + vLLM).
- Automation scripts for running parameter sweeps and collecting power metrics.

## Repo structure

- `dataset/`: scripts for preprocessing ShareGPT data and generating chat-history pickles.
- `src/70BMulti/`: cache simulation + workload driver + automation scripts.

## Prerequisites

- Python 3.10+.
- LMCache + vLLM.

## Dataset preparation

1) Place the ShareGPT V3 JSON at:

```
/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
```

2) Preprocess and add token lengths:

```
python dataset/dataset_preprocessing.py --parse 1
```

3) Generate chat histories and a request sequence:

```
python dataset/dataset_creation.py
```

## End-to-end automation

`src/70BMulti/70BMulti_automation.sh` sweeps cache sizes and lambdas, starts LMCache, runs workloads, and records power.

Run `src/70BMulti/70BMulti_automation.sh` to start the end-to-end automation.

**Warning:** these scripts are destructive and environment-specific. Review paths and commands before running.

