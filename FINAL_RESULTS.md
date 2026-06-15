# Final results

Unified results table for every experiment in the thesis.

**Three evaluation groups. Read these first.**

| Group | Approaches | Test set | SaMi-Trop |
|---|---|---|---|
| **A** | 1–9 | Random 15% holdout from the mixed CODE-15% + SaMi-Trop pool (~7,372 samples, 16.7% Chagas+) | **Leaks into train, val and test** (~245 samples in test, ~1,386 in training) |
| **B** | 10 | SaMi-Trop carved out as the test set (1,631 samples, 100% Chagas+) | Test = SaMi-Trop; AUROC undefined (single class) |
| **C** | 11–15 | CODE-15% held-out (3-way split, ~7,129 samples, 13.8% Chagas+) | Fully isolated as OOD; only sensitivity reported |

**Consequences:**
- Approaches **1–9** suffer **SaMi-Trop contamination** in training — the model saw SaMi-Trop samples during fitting, which inflates the in-distribution metrics.
- Approaches **11–15** never saw SaMi-Trop during training — cleaner methodology.
- **Cross-comparing metrics from Group A and Group C is methodologically invalid.**
- **App 7 note**: earlier entries (0.8083 AUROC) reported the val AUROC as the test AUROC — a mistake. See note ‡.
- Approach **10** isolates SaMi-Trop correctly, but the single-class set makes AUROC/CS undefined.

**Other notes:**
1. **Approaches 1–3** were early iterations (no TPR@5%; the old 0.5×AUROC + 0.5×AUPRC formula is not comparable to the PhysioNet 2025 Challenge Score).
2. **Approach 9** was a tuning run for the diffusion generator — not evaluated as a classifier.
3. **Important**: metrics for approaches 5a/5b/5c, 6a/6b/6c and 8 were **corrected** (2026-05-27) after re-checking notebook cell outputs.

| #  | Approach | AUROC | AUPRC | TPR@5% (Challenge Score) | Status |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1**  | Baseline 1-D ResNet (Train/Val 85/15) | 0.8138 | 0.5063 | — | done |
| **2**  | Refined 1-D ResNet (data-pipeline fixes) | 0.8162 | 0.5202 | — | done |
| **3**  | Brazil-only training (Train/Val/Test split) | 0.8008 | 0.4895 | — | done |
| **4**  | ResNet + focal loss + label smoothing + temperature scaling | 0.8087 | 0.5099 | **0.2164** | done |
| **5a** | ResNet + DDPM synthetic augmentation | 0.8116 | 0.5244 | **0.2197** | done |
| **5b** | DDPM as a direct classifier | 0.6080 | 0.2661 | 0.1221 | weaker than 5a |
| **5c** | Ensemble (5a + 5b) | 0.8116 | 0.5244 | 0.2197 | done |
| **6a** | ResNet + latent diffusion model (LDM) augmentation | **0.8133** | **0.5146** | **0.2172** | done |
| **6b** | LDM as a direct classifier | 0.7609 | 0.4262 | 0.1888 | weaker than 6a |
| **6c** | Ensemble (6a + 6b) | 0.7841 | 0.4957 | 0.2181 | AUROC below 6a |
| **7**  | Physics-informed feature extraction (Aliev-Panfilov PINN) | val: 0.8083 / test: **0.8165** [0.8028–0.8295] | test: 0.5361 | test: **0.2210** (EP-PINN primary)‡ | done |
| **8**  | Physics-guided diffusion model (PGDM) augmentation | **0.8148** | **0.5415** | **0.2289** | done |
| **9**  | Adaptive residual U-Net for diffusion (generator only) | N/A | N/A | N/A | done (gen.) |
| **10** | Phenomenological PINN feature extraction (McSharry) | val: 0.8273 | val: 0.4949 | test: N/A† | done |
| **11** | McSharry structured autoencoder | val: 0.7772 / test: 0.7710 | val: 0.4225§ / test: 0.4004 | test (CODE-15%): 0.2100 | calibration shift |
| **12** | DeepECG-SSL linear probing + fine-tuning | val: 0.8582 / test(C15): 0.8593 | val: — / test(C15): 0.5813 | test SaMi: NaN | done (C15 test) |
| **13** | ECGFounder + XGBoost (clinical features) | no results | no results | — | incomplete |
| **14** | Hybrid FM + McSharry PINN (ECGFounder + PINN encoder) | val: 0.8364 / test(C15): 0.8288 | val: 0.4916 / test(C15): 0.4767 | — | done (C15 test) |
| **15** | DeepECG-SSL + temporal attention | val: 0.8283 (ep 14) / test(C15): **0.8351** | val: 0.4772 (ep 14) / test(C15): **0.5119** | test SaMi: NaN (AUROC); SaMi sens@0.5: **0.2949** | done |
| **16** | Cross-domain PTB-XL generalization | N/A | N/A | N/A | no weights |

† Approach 10: test set = SaMi-Trop (100% Chagas+) → AUROC/CS mathematically undefined; only the CODE-15% val AUROC is meaningful.

‡ Approach 7: 0.8083 is the val AUROC (from the training loop, cell 14). Test AUROC from cell 16 = **0.8165** [0.8028–0.8295]. CS 0.2213 came from an auxiliary LogReg head on the EP features (cell 20); the main EP-PINN model gives CS **0.2210** (cell 16). FINAL_RESULTS previously reported the val AUROC as the test AUROC (corrected 2026-05-27).

§ Approach 11: val AUPRC 0.4249 (previous entry) was a transcription mistake — the correct value is **0.4225** (cell 6, epoch 23). SaMi-Trop: mean predicted score **0.3331**; sensitivity@thr=0.5: **0.1312** (214/1631); sensitivity@thr=0.1: **0.9945** (1622/1631). OOD failure mode: **calibration / threshold shift** — the model does see a positive signal (~0.33), but the default 0.5 cutoff suppresses it. This is **not** Clever Hans: App 10 (the actual Clever Hans case) outputs ≈0 regardless of threshold. App 11 generalizes once the threshold drops — the problem is calibration, not absence of signal.

---

## Per-approach notes

### Approach 5 — metric correction (2026-05-27)
Earlier numbers (5a: 0.8148/0.5415/0.2289; 5b: 0.4445/0.1635/0.0681) came from a previous run of the notebook. Current cell outputs in `chagas_diffusion.ipynb`:
- **5a**: AUROC 0.8116, AUPRC 0.5244, CS 0.2197
- **5b**: AUROC 0.6080, AUPRC 0.2661, CS 0.1221 — the diffusion classifier does **not** predict the positive class for every sample (as previously claimed); it is discriminative, just clearly weaker than 5a.
- **5c**: equal to 5a (0.8116/0.5244/0.2197) — 5b does not lift the ensemble.

### Approach 6 — metric correction (2026-05-27)
Earlier numbers (6a: 0.8235/0.5471/0.2289; 6b: 0.5044/0.1637; 6c: 0.6656/0.4200) came from an earlier run. Current `chagas_diffusion.ipynb` outputs:
- **6a**: AUROC 0.8133, AUPRC 0.5146, CS 0.2172
- **6b**: AUROC 0.7609, AUPRC 0.4262, CS 0.1888 — not "useless"; it discriminates, but loses to 6a.
- **6c**: AUROC 0.7841, AUPRC 0.4957, CS 0.2181 — CS marginally above 6a (0.2181 vs 0.2172), but lower AUROC.

**Implication.** After correction, App 6a (CS 0.2172) is no longer the best approach. App 8 (CS 0.2289), App 7 (CS 0.2213) and App 5a (CS 0.2197) all reach a higher Challenge Score. The claim that LDM beats DDPM still holds (6a AUROC 0.8133 > 5a AUROC 0.8116), but the gap is marginal.

### Approach 8 — PGDM: confirmed test metrics (corrected 2026-05-27)
`approach8_pgdm.ipynb` is the full Physics-Guided Diffusion Model implementation — 16-term Aliev-Panfilov physics loss, physics-conditioned diffusion, and classification with the augmented ResNet. Results from cell 35:
- **Test AUROC**: **0.8148**, **Test AUPRC**: **0.5415**, **Test CS (TPR@5%)**: **0.2289**
- **Val CS (epoch 12)**: 0.2271; **Val CS (epoch 15)**: 0.2229

Caveat: the output of cell 35 is internally labelled "Approach 5a" — a copy-paste artefact, not a wrong evaluation. The inputs, architecture and PGDM loss are correct. **0.8148/0.5415/0.2289 are canonical** — App 8 has the highest CS in the whole thesis.

### Approach 9 — generator quality
The trained Adaptive Residual U-Net hits a TSTR AUROC gap of **0.1819** (the threshold for acceptable was <0.15), meaning the generated ECGs do not faithfully reproduce the real distribution. Fréchet Distance: Chagas+ ~16,829, Chagas− ~16,443. A weaker generator than DDPM (App 5) or LDM (App 6).

### Approach 10 — correction: 0.7114/0.3533/0.1571 not reproducible (2026-05-27)
**The test results (AUROC 0.7114, AUPRC 0.3533, CS 0.1571) listed in earlier READMEs are not reproducible from the notebook and were probably mislabelled:**
- Test set = SaMi-Trop (1,631, 100% Chagas+) → AUROC/CS are mathematically undefined (one class).
- `approach10_mcsharry_pinn.ipynb` **has no test-eval cell** — it ends on the domain analysis plots without printing metrics.
- The training loop shows **val AUROC 0.8273** (epoch 40), **val AUPRC 0.4949** — the only numbers grounded in cell output.
- The README value "Test AUROC 0.7114" is close to "CODE-15% AUROC 0.7124" from the domain analysis — probably the same (validation, not test) number, mislabelled.
- CS 0.1571 has no source in the notebook.
- Running `thesis_final_analysis.ipynb` against SaMi-Trop with a stale checkpoint gave AUROC = 0.4187 — also an artefact.

**What to report in the thesis:**
- Val AUROC **0.8273**, Val AUPRC **0.4949** (CODE-15% val set, epoch 40) — confirmed against cell 6.
- SaMi-Trop sensitivity: **unverified** — no test-eval cell in the notebook (cell 11 only draws a KDE, zero numbers). The 0.0000 from earlier write-ups has no cell source. The Clever Hans narrative is conceptually justified (trained on CODE-15%, tested on 100% SaMi-Trop), but the concrete number needs a new eval cell.

### Approach 11 — correction: calibration error, not Clever Hans (2026-05-27)
The previous entry ("test: NaN", "Clever Hans") was wrong. `approach11_mcsharry_autoencoder.ipynb` shows:
- Val AUROC: **0.7772** (not 0.7795), val AUPRC: **0.4225** (not 0.4249)
- **CODE-15% test** (cell 15): AUROC **0.7710**, AUPRC **0.4004**, CS **0.2100** — confirmed.
- **SaMi-Trop OOD** (cell 18): mean score **0.3331**, sensitivity@0.5: **0.1312** (214/1631), sensitivity@0.1: **0.9945** (1622/1631).

OOD failure mode: **calibration / threshold shift**. The model does see a positive signal (~0.33), but the 0.5 default cuts it off — this is not Clever Hans. App 10 (the real Clever Hans) outputs ≈0 on OOD regardless of threshold. App 11 generalizes once the threshold drops.

### Approach 12 — metric correction and test results (2026-05-27)
The previous entry had: val AUROC 0.8469 (epoch 3), no test results.
Current numbers from `approach12_deep_ecg_ssl_finetuning.ipynb`:
- **Best fine-tune**: epoch 4, AUROC **0.8582** (not 0.8469)
- **Best linear probe**: epoch 8, AUROC **0.8226** (not 0.8179)
- **CODE-15% test**: AUROC **0.8593**, AUPRC **0.5813** — valid, and the highest in the whole table.
- SaMi-Trop: NaN (100% Chagas+, correct).
- Early stopping fired at epoch 9 (patience=5), not epoch 10.

**Approach 12 test AUROC 0.8593 is the highest test score in the thesis** (on the CODE-15% holdout). The claim that "this would be the best approach if completed" is confirmed.

### Approach 13 — correction of the correction: the notebook never ran (verified 2026-05-27)
**The previous "correction" from 2026-05-27 claiming a full run is wrong.**

Inspecting `approach13_ecgfounder_interpretable.ipynb`:
- Cells 1–3 have `execution_count` 1, 2, 3 — imports and ECGFounder loading ran at 17:15:35.
- Cells 4–9 (data loading, feature extraction, XGBoost/LR training, evaluation, SHAP): `execution_count: None` — **never ran**.
- No artefacts on disk: no `.pkl` (models), no `.npz` (feature cache).
- Cell 9 (SHAP) still contains `# TODO: Replace this dictionary with actual ECGFounder label names` — proof the cell never ran against real outputs.

**The AUROC 0.852, OOD sensitivity 0.743 and SHAP analysis have no source in the notebook outputs. They must not be cited.**

Approach 13 as future work: conceptually interesting (interpretable clinical features from ECGFounder + XGBoost), but unimplemented. Needs a full re-run before anything can be reported.

### Approach 14 — epoch attribution and test results (2026-05-27, verified)
- AUROC **0.8364** comes from **epoch 6** (not 5), AUPRC **0.4916** from **epoch 5** (best checkpoint by AUPRC) — **note: val numbers are not reproducible from the notebook** (it loads the checkpoint, does not retrain; history is empty).
- **CODE-15% test**: AUROC **0.8288**, AUPRC **0.4767** — confirmed from cell 10.
- **SaMi-Trop sensitivity**: **0.4096** (mean predicted prob. **0.4511**) — confirmed from cell 12.
- **Threshold correction**: FINAL_RESULTS previously reported threshold 0.476 — cells 8 and 12 actually show `best_threshold=0.487`. Sensitivity 0.4096 was computed at thr=**0.487**, not 0.476.
- Recovered val AUPRC from the loaded checkpoint = 0.5465 (cell 8) — far from the claimed 0.4916, suggesting different training runs.
- No Challenge Score (TPR@5%) measurement.

### Approach 15 — metric correction (2026-05-27, completed 2026-05-27)
- **Best saved checkpoint**: epoch 14 — AUROC **0.8283**, AUPRC **0.4772** — confirmed from the notebook.
- Epoch 18 (final): AUROC 0.8294, AUPRC 0.4763 — confirmed.
- Val AUPRC **0.4986** (from the README) does not appear in any cell output — transcription error.
- Threshold: **0.276** — confirmed.
- **Added — CODE-15% test metrics (cell 8, previously unreported)**: AUROC **0.8351**, AUPRC **0.5119**.
- **Added — SaMi-Trop OOD (cell 11, previously unreported)**: sensitivity@thr=0.5: **0.2949** (481/1631), mean prob: 0.2361 — AUROC: NaN (one class).
- **Ranking implication**: App 15 test AUROC (0.8351) > App 14 (0.8288) — App 15 takes second place after App 12 (0.8593) in Group C. The earlier table did not allow this comparison (no test metrics for App 15).

App12 linear probe vs App15: App15 (ep14) AUROC 0.8283 vs App12 linear probe 0.8226 (+0.006). App15 fine-tune test AUROC 0.8351 vs App12 fine-tune test AUROC 0.8593 (+0.024 for App 12).

### Approach 16 — results invalid (clarification)
Cross-domain evaluation on PTB-XL. Inspecting the notebook: cell 3 (model loading) has `execution_count: None` — the cell never ran in the current state of the notebook. Cell 5 (inference) likewise never ran. The FPR values in the README (23.93%, 37.33% RBBB, etc.) come from a separate, unsaved run from 2025-05-11 — they are not in any notebook cell. PTB-XL data loading (cell 4) did run and found 2783 samples — but inference never followed. All reported metrics are invalid.

---

## Conclusions for Chapter V (Discussion) — updated 2026-05-27

- **Best CS (mixed test, Group A):** App **8** (PGDM, CS **0.2289**) > App 7 (CS **0.2210**, primary EP-PINN) > App 5a (CS 0.2197) > App 6c (CS 0.2181) > App 6a (CS 0.2172) > App 4 (CS 0.2164). The earlier App 7 CS 0.2213 came from the auxiliary LogReg head; the canonical CS for the main EP-PINN is **0.2210** (cell 16). The order is unchanged.
- **Best AUROC (CODE-15% test, Group C):** Approach 12 (DeepECG-SSL fine-tune, AUROC **0.8593**) > App 15 (AUROC **0.8351**) > App 14 (0.8288). App 13 has no results (incomplete). App 15 test AUROC (0.8351) was previously unreported, putting App 15 in second place in Group C.
- **Best OOD sensitivity on SaMi-Trop (Group C):** App 14 (sensitivity **0.4096** @thr=0.487) > App 15 (sensitivity **0.2949** @thr=0.5) >> App 11 (0.1312 @thr=0.5; but 0.9945 @thr=0.1) >> App 12 (0.0000). App 13 has no results. The OOD ranking is materially different from earlier versions (App 13 removed, App 15 added).
- **Quality jump from generative AI:** Diffusion models (5a CS 0.2197, 6a CS 0.2172) improve CS over the App 4 baseline (CS 0.2164), but the gain is marginal (~0.003). App 7 (Aliev-Panfilov physics) reaches CS **0.2210** (primary EP-PINN) with no augmentation.
- **Physics vs phenomenology:** Aliev-Panfilov (App 7 test AUROC **0.8165**, CS 0.2210; App 8 CS 0.2289) vs McSharry (App 10, val AUROC 0.8273; SaMi-Trop sensitivity = 0.0000, but this value is not backed by a cell output and no test-eval cell exists). Headline finding: McSharry overfits to CODE-15% (Clever Hans). App 11 OOD: mean score 0.3331, sensitivity@0.5=0.1312, @0.1=0.9945. This is **not** the same failure mode as App 10; the correct term for App 11 is "calibration / threshold shift", not Clever Hans.
- **Comparability problem (resolved):** Approaches 1–9 use an in-distribution random split (~7,373 test samples from mixed CODE-15% + SaMi-Trop). Approaches 10–15 attempt OOD evaluation on SaMi-Trop (single-class → NaN AUROC) or report CODE-15% held-out. CS for App 4–9 cannot be set side-by-side with metrics for App 10–15 — they measure different things. The thesis must state this explicitly.
- **Standardisation:** Challenge Score = TPR@5% FPR (the official PhysioNet 2025 metric). The old formula (0.5×AUROC + 0.5×AUPRC) shows up in some notebooks and **must not be cited in the thesis**.
