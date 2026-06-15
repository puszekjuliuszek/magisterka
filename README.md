# Chagas detection from 12-lead ECG

Master's thesis code and weights. PhysioNet/CinC Challenge 2025 task: predict serologic
positivity for Chagas disease from a 12-lead resting ECG.

The thesis itself is under `thesis/`. The numbered results table is `FINAL_RESULTS.md`.

## Approaches

Sixteen experiments, grouped by evaluation regime (see `FINAL_RESULTS.md` for the
authoritative numbers and CIs).

| ID  | What                                                       | Notebook                                          |
|-----|------------------------------------------------------------|---------------------------------------------------|
| 1-4 | Baseline 1-D ResNet variants (focal loss, label smoothing) | `notebooks/chagas_resnet_classifier.ipynb`        |
| 5   | ResNet + DDPM synthetic augmentation                       | `notebooks/chagas_diffusion.ipynb`                |
| 6   | ResNet + latent diffusion (LDM) augmentation               | `notebooks/chagas_diffusion.ipynb`                |
| 7   | Aliev-Panfilov PINN, EP features → MLP                     | `notebooks/approach7_pinn_feature_extraction.ipynb` |
| 8   | Physics-guided diffusion (PGDM) augmentation               | `notebooks/approach8_pgdm.ipynb`                  |
| 9   | Adaptive residual U-Net (generator only)                   | `notebooks/approach9_adaptive_unet.ipynb`         |
| 10  | McSharry PINN, 17 ODE features → MLP                       | `notebooks/approach10_mcsharry_pinn.ipynb`        |
| 11  | McSharry structured autoencoder                            | `notebooks/approach11_mcsharry_autoencoder.ipynb` |
| 12  | DeepECG-SSL linear probe + fine-tune                       | `notebooks/approach12_deep_ecg_ssl_finetuning.ipynb` |
| 13  | ECGFounder + XGBoost (incomplete)                          | `notebooks/approach13_ecgfounder_interpretable.ipynb` |
| 14  | Hybrid ECGFounder + McSharry PINN                          | `notebooks/approach14_hybrid_fm_pinn.ipynb`       |
| 15  | DeepECG-SSL + temporal attention                           | `notebooks/approach15_ecg_fm_attention.ipynb`     |
| 16  | Cross-domain check on PTB-XL                               | `notebooks/approach16_cross_domain_generalization.ipynb` |

Failure-mode analysis and per-subject FP/FN figures live in
`notebooks/thesis_final_analysis.ipynb`.

## Datasets

- CODE-15% (Brazil) - training, validation, held-out test (3-way stratified split, seed 42).
- SaMi-Trop (Brazil, 100% Chagas+) - out-of-distribution test in the Group-C protocol
  (approaches 11, 12, 14, 15).
- PTB-XL (Germany, ~0% prevalence) - cross-domain check in approach 16.

Datasets are not redistributed here; download them from PhysioNet and PTB-XL and rebuild
the cache by running `notebooks/chagas_resnet_classifier.ipynb` once.

## Repo layout

```
notebooks/   one .ipynb per approach
weights/     *_best.pt and *_last.pt checkpoints (>100 MB; Git LFS recommended)
src/         shared building blocks (ResNet, dataset, trainer, metrics, preprocessing)
configs/     baseline YAML
thesis/      LaTeX sources for the thesis (main.tex, tex/, img/, bibliography.bib)
FINAL_RESULTS.md  authoritative results table (Polish)
```

## Running

Python 3.10+, PyTorch 2.x. Apple Silicon (`mps`) and CUDA both supported by the notebooks.

```
pip install -r requirements.txt
```

Run `notebooks/chagas_resnet_classifier.ipynb` first - it builds `preprocessed_cache.h5`
and `preprocessed_cache_brazil.h5` that every other notebook reuses.

For approaches 12/14/15/16 you also need DeepECG-SSL / ECGFounder weights from their
respective upstream repositories; paths are set in the `CFG` dict at the top of each
notebook.

## Weights

Files under `weights/` follow the convention `approach<N>_<variant>_best.pt`
(best validation metric) or `..._last.pt` (final epoch). Several are larger than
GitHub's 100 MB per-file limit; if you clone this repo you will want
[Git LFS](https://git-lfs.com/) configured for `*.pt`.

Intermediate epoch checkpoints from training were not committed.

## License

See `LICENSE`.
