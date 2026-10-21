# Reproduced results

All metrics below are recomputed from released per-example outcomes.
Macro P/R/F1 and accuracy are fractions in CSV; MAE/MSE are unnormalized ordinal errors.

## Paired VLM results

| Method | Condition | Task | P (%) | R (%) | F1 (%) | Acc (%) | MAE | MSE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| gpt56 | masked | bias | 73.078 | 60.038 | 56.729 | 64.662 | 0.444 | 0.684 |
| gpt56 | masked | factuality | 71.183 | 71.304 | 69.320 | 70.677 | 0.293 | 0.293 |
| kimi | masked | bias | 64.845 | 59.600 | 55.774 | 64.662 | 0.414 | 0.579 |
| kimi | masked | factuality | 71.309 | 71.826 | 70.880 | 71.429 | 0.301 | 0.331 |
| ministral_base | masked | bias | 27.546 | 31.979 | 26.838 | 26.316 | 0.977 | 1.624 |
| ministral_base | masked | factuality | 50.102 | 41.434 | 34.659 | 40.602 | 0.654 | 0.774 |
| ministral_finetuned | masked | bias | 68.887 | 69.526 | 68.300 | 73.684 | 0.383 | 0.744 |
| ministral_finetuned | masked | factuality | 79.748 | 75.915 | 76.103 | 75.940 | 0.248 | 0.263 |
| opus5 | masked | bias | 68.980 | 61.512 | 62.938 | 66.917 | 0.466 | 0.932 |
| opus5 | masked | factuality | 74.461 | 74.642 | 74.529 | 74.436 | 0.256 | 0.256 |
| qwen | masked | bias | 55.276 | 40.189 | 41.221 | 40.602 | 1.353 | 4.180 |
| qwen | masked | factuality | 67.554 | 55.720 | 55.875 | 54.887 | 0.564 | 0.789 |
| qwen35_base | masked | bias | 34.586 | 42.297 | 37.015 | 48.872 | 0.789 | 1.752 |
| qwen35_base | masked | factuality | 57.558 | 40.984 | 30.194 | 40.602 | 0.857 | 1.383 |
| qwen35_finetuned | masked | bias | 75.516 | 72.886 | 72.255 | 77.444 | 0.338 | 0.669 |
| qwen35_finetuned | masked | factuality | 82.265 | 81.216 | 81.537 | 81.203 | 0.195 | 0.211 |
| gpt56 | unmasked | bias | 72.413 | 62.611 | 60.677 | 66.917 | 0.406 | 0.602 |
| gpt56 | unmasked | factuality | 71.212 | 71.203 | 69.977 | 70.677 | 0.293 | 0.293 |
| kimi | unmasked | bias | 65.048 | 59.380 | 56.131 | 64.662 | 0.414 | 0.579 |
| kimi | unmasked | factuality | 71.457 | 71.761 | 71.379 | 71.429 | 0.286 | 0.286 |
| ministral_base | unmasked | bias | 30.645 | 35.029 | 29.650 | 28.571 | 0.895 | 1.376 |
| ministral_base | unmasked | factuality | 51.412 | 42.914 | 37.920 | 42.105 | 0.662 | 0.827 |
| ministral_finetuned | unmasked | bias | 75.525 | 73.254 | 72.185 | 77.444 | 0.338 | 0.684 |
| ministral_finetuned | unmasked | factuality | 81.342 | 79.756 | 79.526 | 79.699 | 0.226 | 0.271 |
| opus5 | unmasked | bias | 71.600 | 67.674 | 68.518 | 72.180 | 0.346 | 0.571 |
| opus5 | unmasked | factuality | 84.030 | 83.471 | 83.640 | 83.459 | 0.165 | 0.165 |
| qwen | unmasked | bias | 52.963 | 41.435 | 40.796 | 42.857 | 1.263 | 3.805 |
| qwen | unmasked | factuality | 67.303 | 58.719 | 57.522 | 57.895 | 0.511 | 0.692 |
| qwen35_base | unmasked | bias | 36.498 | 41.270 | 36.246 | 46.617 | 0.895 | 2.143 |
| qwen35_base | unmasked | factuality | 64.451 | 48.736 | 38.964 | 48.120 | 0.707 | 1.083 |
| qwen35_finetuned | unmasked | bias | 80.686 | 74.455 | 73.707 | 78.195 | 0.361 | 0.797 |
| qwen35_finetuned | unmasked | factuality | 80.862 | 79.804 | 79.864 | 79.699 | 0.211 | 0.226 |

## Coverage

All eight VLM methods are included for both tasks on the main test and both paired conditions.
See provenance/coverage.json and docs/PROTOCOLS.md for source details.
Reference metrics preserve V1 Opus/GPT and V2 Kimi/Qwen coding; comparisons remain rubric-dependent.
Full 133-pair adapter scores include 43 training-seen outlets (42 identical unmasked training images).
The 90 unseen-outlet scores are included separately in metrics.csv.
