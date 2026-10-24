"""
main.py
========
NPOL Tonal Detection Suite -- single entry point.

    python main.py

launches an interactive menu that dispatches to every module. Each
option runs the corresponding script as a subprocess (so the heavy
imports for, say, the Tkinter GUI or PyTorch training loop are only
paid for by the option you actually pick), and every module reads its
tunable parameters from `config.yaml` (edit directly, or use option 9).
"""
from __future__ import annotations
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, "13_Scripts"))
sys.path.append(os.path.join(ROOT, "09_Models"))
sys.path.append(os.path.join(ROOT, "04_Signal_Generator"))
sys.path.append(os.path.join(ROOT, "05_Dataset_Generator"))
from config_loader import load_config, get as cfg_get, save_config  # noqa: E402

PY = sys.executable

BANNER = r"""
=================================================
 NPOL TONAL DETECTION SUITE
=================================================
 1. Train Model
 2. Test Model
 3. Live Detection
 4. Signal Generator
 5. Dataset Generator
 6. Model Evaluation
 7. Deployment
 8. Ethernet Receiver
 9. Settings
 10. Exit
=================================================
"""


def run(cmd: list[str], cwd: str | None = None):
    print(f"\n$ {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, cwd=cwd, check=False)
    except KeyboardInterrupt:
        print("\n[main] interrupted.")
    except FileNotFoundError as e:
        print(f"[main] could not launch: {e}")


def prompt(text: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"{text}{suffix}: ").strip()  # raises EOFError on piped/closed stdin -- caught in main()
    return val or default


# --------------------------------------------------------------------- menu
def option_train():
    model = prompt("Model to train (segformer/unet)", "segformer")
    epochs = prompt("Epochs", "20")
    steps = prompt("Steps per epoch", "100")
    batch = prompt("Batch size", "8")
    real_dir = prompt("Real-data dir for noise injection (blank = synthetic only)", "")
    cmd = [PY, "train.py", "--model", model, "--epochs", epochs,
           "--steps-per-epoch", steps, "--batch-size", batch]
    if real_dir:
        cmd += ["--real-data-dir", real_dir]
    run(cmd, cwd=os.path.join(ROOT, "01_Training_Module"))


def option_test():
    run([PY, "-m", "pytest", "-v"], cwd=os.path.join(ROOT, "02_Testing_Module"))


def option_live_detection():
    print("\nModel source:")
    print("  1. Use active model from config.yaml")
    print("  2. Pick interactively from 11_Weights/ (Model Manager)")
    choice = prompt("Choice", "1")
    src = prompt("Input source (synthetic/wav/ethernet)", "synthetic")
    cmd = [PY, "live_detector.py", "--source", src]
    if choice == "2":
        cmd.append("--model-manager")
    if src == "synthetic":
        snr = prompt("Target SNR (dB)", "-8")
        cmd += ["--snr-db", snr]
    elif src == "wav":
        wav = prompt("Path to WAV file")
        cmd += ["--wav-path", wav]
    run(cmd, cwd=os.path.join(ROOT, "03_Live_Detector"))


def option_signal_generator():
    from waveforms import WAVEFORM_TYPES  # noqa: local import, only needed here
    print(f"\nWaveform types: {', '.join(WAVEFORM_TYPES)}")
    wtype = prompt("Type", "multi_tone")
    snr = prompt("SNR (dB, range +20..-15)", "-8")
    duration = prompt("Duration (s)", "4")
    cmd = [PY, "generator_cli.py", "--type", wtype, "--snr-db", snr, "--duration", duration]
    run(cmd, cwd=os.path.join(ROOT, "04_Signal_Generator"))


def option_dataset_generator():
    n = prompt("Number of samples to materialize", "200")
    out_dir = prompt("Output dir", os.path.join(ROOT, "10_Datasets", "materialized"))
    run([PY, "-c",
         f"import dataset_builder as d; d.materialize_dataset({out_dir!r}, n_samples={int(n)})"],
        cwd=os.path.join(ROOT, "05_Dataset_Generator"))


def option_model_evaluation():
    n = prompt("Samples per SNR point", "6")
    entry_path = _select_checkpoint_or_default()
    run([PY, "evaluate.py", "--checkpoint", entry_path, "--n-per-snr", n],
        cwd=os.path.join(ROOT, "06_Model_Evaluation"))


def option_deployment():
    print("\n  1. Launch GUI with best available deployment artifact")
    print("  2. Launch GUI, pick model explicitly")
    print("  3. Headless (console-only) run")
    choice = prompt("Choice", "1")
    if choice == "2":
        path = _select_checkpoint_or_default(required=True)
        run([PY, "deploy.py", "--checkpoint", path], cwd=ROOT)
    elif choice == "3":
        n = prompt("Number of chunks", "20")
        run([PY, "deploy.py", "--headless", "--n-chunks", n], cwd=ROOT)
    else:
        run([PY, "deploy.py"], cwd=ROOT)


def option_ethernet_receiver():
    print("\n  1. Start receiver (listen for incoming stream)")
    print("  2. Start sender simulator (test locally, needs receiver or Live Detector running)")
    choice = prompt("Choice", "1")
    if choice == "2":
        snr = prompt("SNR (dB)", "-8")
        run([PY, "ethernet_sender_sim.py", "--mode", "synthetic", "--snr-db", snr],
            cwd=os.path.join(ROOT, "08_Ethernet_Receiver"))
    else:
        run([PY, "receiver.py"], cwd=os.path.join(ROOT, "08_Ethernet_Receiver"))


def option_settings():
    cfg = load_config(force_reload=True)
    while True:
        print("\n--- Settings (config.yaml) ---")
        print(f"  1. Sampling rate:          {cfg_get(cfg, 'signal.sampling_rate_hz')}")
        print(f"  2. FFT size:               {cfg_get(cfg, 'signal.fft_size')}")
        print(f"  3. Hop length:             {cfg_get(cfg, 'signal.hop_length')}")
        print(f"  4. Band-pass (low/high Hz):{cfg_get(cfg, 'signal.bandpass.low_hz')} / "
              f"{cfg_get(cfg, 'signal.bandpass.high_hz')}")
        print(f"  5. Confidence threshold:   {cfg_get(cfg, 'model.confidence_threshold')}")
        print(f"  6. CFAR Pfa:               {cfg_get(cfg, 'cfar.pfa')}")
        print(f"  7. Active model:           {cfg_get(cfg, 'model.active')}")
        print(f"  8. List available models (Model Manager)")
        print(f"  9. Open config.yaml path")
        print(f"  0. Back to main menu")
        choice = prompt("Choice", "0")
        if choice == "0":
            return
        elif choice == "8":
            from model_manager import list_models_pretty
            print(list_models_pretty())
        elif choice == "9":
            print(f"  config.yaml -> {os.path.join(ROOT, 'config.yaml')}")
        elif choice in {"1", "2", "3", "4", "5", "6", "7"}:
            key_map = {
                "1": "signal.sampling_rate_hz", "2": "signal.fft_size", "3": "signal.hop_length",
                "5": "model.confidence_threshold", "6": "cfar.pfa", "7": "model.active",
            }
            if choice == "4":
                low = prompt("New low_hz", str(cfg_get(cfg, "signal.bandpass.low_hz")))
                high = prompt("New high_hz", str(cfg_get(cfg, "signal.bandpass.high_hz")))
                cfg["signal"]["bandpass"]["low_hz"] = float(low)
                cfg["signal"]["bandpass"]["high_hz"] = float(high)
            else:
                key = key_map[choice]
                cur = cfg_get(cfg, key)
                new_val = prompt(f"New value for {key}", str(cur))
                node = cfg
                parts = key.split(".")
                for p in parts[:-1]:
                    node = node[p]
                # keep the original type (int/float/str) where possible
                try:
                    node[parts[-1]] = type(cur)(new_val) if cur is not None else new_val
                except (TypeError, ValueError):
                    node[parts[-1]] = new_val
            save_config(cfg)
            print("  saved.")
        else:
            print("  invalid choice.")


def _select_checkpoint_or_default(required: bool = False) -> str:
    from model_manager import pick_model_interactive
    use_picker = prompt("Pick a model interactively? (y/N)", "n").lower().startswith("y")
    if use_picker or required:
        entry = pick_model_interactive()
        if entry:
            return entry.path
    cfg = load_config()
    return os.path.join(ROOT, cfg_get(cfg, "model.segformer_weights", "11_Weights/segformer_b0_tonal.pt"))


MENU = {
    "1": option_train,
    "2": option_test,
    "3": option_live_detection,
    "4": option_signal_generator,
    "5": option_dataset_generator,
    "6": option_model_evaluation,
    "7": option_deployment,
    "8": option_ethernet_receiver,
    "9": option_settings,
}


def main():
    while True:
        print(BANNER)
        try:
            choice = input("Select an option [1-10]: ").strip()
        except EOFError:
            print("\nGoodbye.")
            return
        if choice in ("10", "exit", "quit", "q"):
            print("Goodbye.")
            return
        action = MENU.get(choice)
        if action is None:
            print("Invalid choice, try again.")
            continue
        try:
            action()
        except KeyboardInterrupt:
            print("\n[main] cancelled.")
        except EOFError:
            print("\nGoodbye.")
            return
        except Exception as e:  # noqa: BLE001 -- keep the launcher alive on module errors
            print(f"[main] error running option {choice}: {e}")
        try:
            input("\nPress Enter to return to the menu...")
        except EOFError:
            print("\nGoodbye.")
            return


if __name__ == "__main__":
    main()
