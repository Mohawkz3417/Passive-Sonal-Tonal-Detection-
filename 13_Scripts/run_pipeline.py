"""
run_pipeline.py
==================
One entry point for the whole generate -> train -> evaluate -> validate ->
demonstrate workflow. This is a DISPATCHER, not a reimplementation: every
flag below subprocess-calls a script that already exists and is already
independently testable/runnable on its own -- this file just saves you
from having to remember which folder each one lives in.

    python 13_Scripts/run_pipeline.py --health              # system_health_check.py --quick
    python 13_Scripts/run_pipeline.py --health --full       # system_health_check.py --full
    python 13_Scripts/run_pipeline.py --check-training-data # 01_Training_Module/check_training_data.py
    python 13_Scripts/run_pipeline.py --train-smoke         # train.py, 1 epoch / 5 steps -- WIRING TEST ONLY
    python 13_Scripts/run_pipeline.py --train               # prints (does NOT run) the exact training command
    python 13_Scripts/run_pipeline.py --evaluate            # 06_Model_Evaluation/evaluate.py
    python 13_Scripts/run_pipeline.py --real-validation     # 06_Model_Evaluation/run_real_validation.py --smoke
    python 13_Scripts/run_pipeline.py --full-test           # tests/ (pytest) + system_health_check.py --full
    python 13_Scripts/run_pipeline.py --generate-data       # 05_Dataset_Generator/dataset_builder.py materialize_dataset

--train deliberately never executes training itself (see ABSOLUTE RULES
in the freeze spec this file was built against) -- it prints the exact
command, resolved from your current config.yaml, for you to run yourself.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "13_Scripts"))
from config_loader import load_config, get as cfg_get  # noqa: E402


def run(cmd: list, cwd: str | None = None) -> int:
    print(f"$ {' '.join(cmd)}" + (f"   (cwd={cwd})" if cwd else ""))
    return subprocess.call(cmd, cwd=cwd or ROOT)


def cmd_health(full: bool) -> int:
    cmd = [sys.executable, os.path.join(ROOT, "13_Scripts", "system_health_check.py")]
    if full:
        cmd.append("--full")
    return run(cmd)


def cmd_check_training_data(n_samples: int) -> int:
    return run([sys.executable, os.path.join(ROOT, "01_Training_Module", "check_training_data.py"),
                "--n-samples", str(n_samples)])


def cmd_generate_data(n_samples: int) -> int:
    script = os.path.join(ROOT, "05_Dataset_Generator", "dataset_builder.py")
    return run([sys.executable, "-c",
                f"import sys; sys.path.insert(0,'{os.path.join(ROOT, '05_Dataset_Generator')}'); "
                f"from dataset_builder import materialize_dataset; "
                f"materialize_dataset('{os.path.join(ROOT, '10_Datasets', 'demo_samples')}', n_samples={n_samples})"])


def _train_command(cfg: dict) -> list:
    out = cfg_get(cfg, "training.output_checkpoint", "11_Weights/active/segformer_b0_tonal.pt")
    return [
        sys.executable, "train.py",
        "--model", str(cfg_get(cfg, "training.model", "segformer")),
        "--epochs", str(cfg_get(cfg, "training.epochs", 50)),
        "--steps-per-epoch", str(cfg_get(cfg, "training.steps_per_epoch", 200)),
        "--batch-size", str(cfg_get(cfg, "training.batch_size", 8)),
        "--lr", str(cfg_get(cfg, "training.lr", 3e-4)),
        "--duration", str(cfg_get(cfg, "training.duration_s", 4.0)),
        "--snr-lo", str(cfg_get(cfg, "training.snr_lo_db", -12.0)),
        "--snr-hi", str(cfg_get(cfg, "training.snr_hi_db", 2.0)),
        "--hard-snr-prob", str(cfg_get(cfg, "training.hard_snr_prob", 0.7)),
        "--hard-snr-lo", str(cfg_get(cfg, "training.hard_snr_range_db", [-12.0, -7.0])[0]),
        "--hard-snr-hi", str(cfg_get(cfg, "training.hard_snr_range_db", [-12.0, -7.0])[1]),
        "--out", os.path.join(ROOT, out),
    ]


def cmd_train(cfg: dict) -> int:
    cmd = _train_command(cfg)
    print("DO NOT TRAIN is in effect for this dispatcher -- the exact command, resolved from your")
    print("current config.yaml, is printed below. Run it yourself when you're ready:\n")
    print(" ".join(cmd))
    print(f"\n(run from: {os.path.join(ROOT, '01_Training_Module')})")
    return 0


def cmd_train_smoke(cfg: dict) -> int:
    print("--train-smoke: 1 epoch x 5 steps, batch 4 -- WIRING TEST ONLY, not real training, "
          "does not touch your active checkpoint (writes to a throwaway path).\n")
    throwaway = os.path.join(ROOT, "11_Weights", "archive", "_smoke_test_checkpoint.pt")
    cmd = [sys.executable, "train.py", "--model", str(cfg_get(cfg, "training.model", "segformer")),
           "--epochs", "1", "--steps-per-epoch", "5", "--batch-size", "4", "--out", throwaway]
    rc = run(cmd, cwd=os.path.join(ROOT, "01_Training_Module"))
    if rc == 0 and os.path.exists(throwaway):
        os.remove(throwaway)
        hist = os.path.splitext(throwaway)[0] + "_history.json"
        if os.path.exists(hist):
            os.remove(hist)
        print("smoke checkpoint verified end-to-end (dataset -> dataloader -> model -> loss -> "
              "backward -> validation -> checkpoint save) and cleaned up.")
    return rc


def cmd_evaluate() -> int:
    return run([sys.executable, "evaluate.py"], cwd=os.path.join(ROOT, "06_Model_Evaluation"))


def cmd_real_validation() -> int:
    return run([sys.executable, "run_real_validation.py", "--smoke"],
               cwd=os.path.join(ROOT, "06_Model_Evaluation"))


def cmd_full_test() -> int:
    rc1 = run([sys.executable, "-m", "pytest", "-v", "."], cwd=os.path.join(ROOT, "tests"))
    rc2 = run([sys.executable, "-m", "pytest", "-v", "."], cwd=os.path.join(ROOT, "02_Testing_Module"))
    rc3 = cmd_health(full=True)
    return rc1 or rc2 or rc3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--health", action="store_true")
    ap.add_argument("--full", action="store_true", help="use with --health for the --full checks")
    ap.add_argument("--check-training-data", action="store_true")
    ap.add_argument("--n-samples", type=int, default=300, help="for --check-training-data / --generate-data")
    ap.add_argument("--generate-data", action="store_true")
    ap.add_argument("--train", action="store_true", help="prints the training command, does NOT run it")
    ap.add_argument("--train-smoke", action="store_true", help="tiny wiring-only run, not real training")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--real-validation", action="store_true")
    ap.add_argument("--full-test", action="store_true")
    args = ap.parse_args()

    if not any([args.health, args.check_training_data, args.generate_data, args.train,
                args.train_smoke, args.evaluate, args.real_validation, args.full_test]):
        ap.print_help()
        return 0

    cfg = load_config()
    rc = 0
    if args.health:
        rc = cmd_health(args.full) or rc
    if args.check_training_data:
        rc = cmd_check_training_data(args.n_samples) or rc
    if args.generate_data:
        rc = cmd_generate_data(args.n_samples) or rc
    if args.train:
        rc = cmd_train(cfg) or rc
    if args.train_smoke:
        rc = cmd_train_smoke(cfg) or rc
    if args.evaluate:
        rc = cmd_evaluate() or rc
    if args.real_validation:
        rc = cmd_real_validation() or rc
    if args.full_test:
        rc = cmd_full_test() or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
