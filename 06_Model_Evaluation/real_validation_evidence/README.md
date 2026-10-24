# Real dataset validation evidence (this suite's own session, pre-packaging)

These files are the genuine output of actually running this project's code
against real ShipsEar / DS3500 / NOAA SanctSound ocean-ambient recordings
before this package was finalized -- not samples, not illustrations. See
`TRAINING_AND_EVAL_REPORT.md` section 7 for the narrative write-up these
files back up.

- `{shipsear,ds3500,ocean_ambient}_results.csv` / `_summary.json` -- output
  of `06_Model_Evaluation/run_real_validation.py --dataset <name>
  --files-per-dataset 30`, one row per real file (file paths shown
  relative to that dataset's own folder).
- `live_real_background_results.json` -- output of a live ground-truth
  test: a known 440 Hz synthetic tonal mixed onto each real dataset's
  background via `/api/real/generate`, transmitted through the real UDP
  array pipeline, and compared against the known true frequency/bearing.

This folder ships as evidence; it is not regenerated automatically. To
reproduce it yourself once you've added real dataset files under
`10_Datasets/real/<name>/` (see that folder's README), re-run the same
commands -- the live script additionally needs both backend services
running (`detection-backend-app` on :8000, `signal-generator-app` on :8001).
