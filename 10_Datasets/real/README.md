# Real dataset folders (bring your own data)

This project does **not** ship any third-party audio datasets -- only the
folder structure, config wiring, and instructions to plug your own copies
in. Drop extracted dataset files into the matching folder below (any
internal sub-folder layout is fine; files are found recursively):

- `shipsear/`       -- ShipsEar underwater ship/ambient noise database (public)
- `ds3500/`         -- DS3500 underwater acoustic dataset
- `ocean_ambient/`  -- NOAA SanctSound-derived ocean ambient noise (animal /
                        anthropogenic / sonar clips)

Once files are in place, `config.yaml`'s `datasets:` section already
points at these folders by relative path, so no further configuration is
needed. Verify with:

```bash
python 13_Scripts/system_health_check.py --full
# or just list what's configured:
curl http://127.0.0.1:8000/api/real_datasets
```

## What these folders are used for

1. **Real-background validation** -- `06_Model_Evaluation/run_real_validation.py
   --dataset shipsear --files-per-dataset 30` runs the trained detector
   against real recordings and writes an honest CSV/JSON report to
   `06_Model_Evaluation/real_validation/`.
2. **Real-noise training** -- pass `--real-data-dir 10_Datasets/real/<name>`
   to `01_Training_Module`'s training script (see `training.real_data_dir`
   / `training.real_noise_prob` in `config.yaml`) to mix real recordings in
   as training-time noise backgrounds instead of only synthetic noise.
3. **Live "real background" demo** -- the Signal Generator's
   `/api/real/generate` endpoint and the frontend's "Real Background" mode
   mix a synthetic, ground-truth tonal onto a real recording from one of
   these folders, so you can see/hear the detector working against
   genuine ocean/ship noise with a known-correct answer to check against.

## Honesty note (read before trusting real-background DOA results)

Public single-channel datasets (ShipsEar/DS3500/etc.) have no real
8-hydrophone array capture. The real-background demo approximates an
array by giving each simulated channel an independently time-shifted
crop of the same mono recording (see
`13_Scripts/real_audio_mixing.py::mix_real_background_per_channel`) --
this is clearly labeled in the API response as approximated, and bearing
accuracy under real backgrounds at low SNR is measurably worse than
under fully synthetic noise for that reason. This is a known, documented
limitation of using mono third-party recordings for array simulation --
not a bug in the detector itself. See `TRAINING_AND_EVAL_REPORT.md` ->
"Real Dataset Validation" for the measured numbers.
