# Failure-example photographs

Three original JPEGs from [birder-project/CUB_200_2011](https://huggingface.co/datasets/birder-project/CUB_200_2011), the dataset used by `src/load_data.py`. Retrieved on 3 November 2026 by reading exact `__key__` matches and extracting `jpg.bytes` from the Hugging Face Parquet conversion. No generated substitutes or image edits were used. Original photographer credits embedded in the photos are preserved.

`sources.json` records the exact image IDs, download sources, row groups/positions and SHA-256 hashes. IDs beginning `validation/` are the mirror's official test images, not the project's validation set. The photos illustrate the rows in `../test_failure_examples.csv`.

These are full source photos. Evaluation uses RGB conversion, resize of the shorter side to 256 and a 224×224 center crop before ImageNet normalization. The report distinguishes visible observations from hypotheses about model errors; no saliency or causal ablation was performed.

Only these three photos were downloaded into the repository. The full processed dataset is still absent; the normal loader would create `data/processed/cub_200_2011`.
