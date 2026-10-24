# Training With Your Real Datasets (ShipsEar, DeepShip, DS3500, ...)

You have real recordings. Here's exactly how they plug into this suite,
what they can and can't give you automatically, and how to use them.

## The core issue: real datasets give you audio, not tonal labels

ShipsEar, DeepShip, DS3500, and similar public datasets give you real
recordings (often with a ship-class label like "tanker" or "passenger
ship" for the whole file), but essentially none of them give you
**per-frequency, per-time-frame ground truth of exactly where a tonal
line sits in the spectrogram** -- that's specialist sonar-analyst
annotation work that these datasets don't include. So you can't just
point the existing `TonalSegDataset` at a folder of these files the way
it works with synthetic data (which knows the ground truth because it
generated the tonal itself).

This suite gives you **two complementary ways** to use your datasets
anyway, and you should use both:

## Method 1 (primary, no manual work): Real-noise injection

`05_Dataset_Generator/dataset_builder.py`'s `TonalSegDataset` now accepts
`real_noise_dir`. When set, a fraction of training samples (`real_noise_prob`,
default 0.5) use a **real recording as the background noise**, with a
**synthetic tonal mixed on top at a controlled SNR** -- so you get:
- real ocean ambient noise statistics, real ship self-noise/machinery
  broadband noise, real sensor characteristics
- exact, free ground truth (you injected the tonal, so you know precisely
  where it is) -- no labeling effort at all

This is the single highest-value thing your datasets give you, and it's
already wired into `train.py`:

```bash
cd 01_Training_Module
python train.py --model segformer --epochs 50 --steps-per-epoch 200 --batch-size 8 \
    --real-data-dir /path/to/ShipsEar \
    --real-noise-prob 0.6 \
    --out ../11_Weights/segformer_b0_tonal.pt
```

`--real-data-dir` accepts **one folder** -- if you want to mix multiple
datasets (ShipsEar + DeepShip + DS3500 + an underwater-noise-only set),
either:
- point it at a parent folder containing all of them as subfolders
  (the loader scans recursively, any layout is fine), or
- run several short training phases, one per `--real-data-dir`, loading
  the previous checkpoint each time (see "resuming/fine-tuning" below).

`05_Dataset_Generator/real_data_loader.py` (`RealAudioPool`) is the
piece doing this: it recursively indexes every `.wav`/`.flac` file under
the folder you give it, and serves random resampled crops. It's robust
to a few corrupt files (skips and warns rather than crashing) and loops
short files to fill the requested clip length.

## Method 2 (small, human-reviewed set): Pseudo-labeling real tonals

Some of your recordings likely contain *actual* ship tonals. To get a
small set of genuinely-real (not synthetic-on-real-noise) training
examples, use:

```bash
cd 05_Dataset_Generator
python pseudo_label_real_tonals.py --input /path/to/ShipsEar \
    --out-dir ../10_Datasets/pseudo_labeled --persistence 0.6
```

This runs a classical, high-recall detector (CFAR + a persistence check
requiring a bin to be flagged across most of a rolling time window --
real tonals are persistent, one-off noise spikes aren't) over every file
and writes:
- `<name>_<chunk>.npz` -- candidate `(spectrogram, mask)` pairs, same
  format as `make_sample()` output
- `<name>_<chunk>_review.png` -- **the spectrogram with the candidate
  mask outlined in cyan, for you to visually check**

**Always open the review PNGs before trusting these as ground truth.**
This detector is tuned for recall (it will include some false positives
-- e.g. a loud transient that happens to persist briefly). Delete or fix
the `.npz` files for candidates that look wrong. Treat what's left as a
small, real-labeled fine-tuning set, and mix it into training as a
second phase after the bulk synthetic+real-noise training above (a short
fine-tuning run at a low learning rate, e.g. `--lr 5e-5 --epochs 5`,
loading `.npz` samples from `10_Datasets/pseudo_labeled/` -- a minimal
`Dataset` for exactly this is a natural next addition to `train.py` if
you want to automate it further; for now, `evaluate.py`'s
`pixel_metrics_np` function and the `.npz` format make it straightforward
to write a short custom loop for this final fine-tuning phase).

## Which datasets do what

| Dataset | Good for |
|---|---|
| ShipsEar | Real-noise injection (Method 1); some files have distinct machinery tonals worth pseudo-labeling (Method 2) |
| DeepShip | Same as above; larger, more vessel-class variety |
| DS3500 | Same as above |
| "Underwater Noise Dataset" (pure ambient, no ships) | Especially valuable for Method 1's **negative** (no-tonal) examples -- real ambient noise with nothing to detect, which teaches the model what to *not* flag |

## Will it "just work" afterwards?

Realistically: it will work much better than the synthetic-only
checkpoint shipped in this delivery (see `RESULTS.md` for why that one
isn't production-ready yet), but budget for:
1. **A full training run** (thousands of steps -- see `SETUP.md`
   Section 3), now with `--real-data-dir` set.
2. **Re-running `06_Model_Evaluation/evaluate.py`** to get real Pd/Pfa-vs-SNR
   numbers on synthetic-on-real-noise test data.
3. **Manually spot-checking on a held-out chunk of your real recordings**
   (run them through `07_Deployment_Module/inference_pipeline.py`
   directly, or through the Live GUI with `--source wav`) since the
   automated evaluation harness still measures against synthetic tonals
   -- your own eyes on real spectrograms are the real acceptance test.
4. If accuracy on real data still lags, increase `real_noise_prob`
   toward 0.8-1.0 and/or invest more time in Method 2's pseudo-labeled
   set -- that's the standard "domain gap" fix.

## Ethernet vs. "from IP" -- these are (mostly) the same thing here

Your hydrophone/acquisition system almost certainly talks over Ethernet
using IP (UDP or TCP) -- "Ethernet" just names the physical cable/link,
the actual data path is IP. `08_Ethernet_Receiver/` already supports
both, and both work from **any reachable IP**, not just localhost:

- **UDP** (`receiver.py`, `EthernetReceiver`) -- listens for datagrams
  pushed to it. Lowest latency; if a packet is lost on a bad link, that
  chunk of audio is simply missing (fine for continuous monitoring,
  not fine if you need every sample guaranteed).
  ```bash
  python live_detector.py --source ethernet --eth-protocol udp \
      --eth-host 0.0.0.0 --eth-port 50000
  ```
  `--eth-host 0.0.0.0` means "listen on all of this machine's network
  interfaces" -- so it will receive from the acquisition system whether
  it's on the same machine, the same LAN, or a routed remote IP, as long
  as the acquisition system is configured to *send to* this laptop's IP
  and the port isn't blocked by a firewall.

- **TCP** (`receiver.py`, `EthernetReceiverTCP`, new in this update) --
  connects *out* to the acquisition system as a client. Use this if your
  hardware runs as a TCP server waiting for client software to connect
  (a common pattern for lab acquisition boxes), or if you need
  guaranteed, ordered delivery over a lossy link.
  ```bash
  python live_detector.py --source ethernet --eth-protocol tcp \
      --eth-host 192.168.1.50 --eth-port 50000
  ```
  Here `--eth-host` is the **acquisition system's IP address** (not this
  laptop's) -- the semantics flip between UDP (you listen) and TCP (you
  connect out), see the `--help` text.

Both share the exact same wire packet format (`pack_packet`/
`unpack_packet` in `receiver.py`), documented in that file. If your real
hardware speaks a different native protocol, you only need to adapt
`unpack_packet()` (or write an equivalent parser) -- nothing downstream
(preprocessing, model, GUI) needs to change either way.

Test either path with no real hardware using the simulator:
```bash
cd 08_Ethernet_Receiver
python ethernet_sender_sim.py --mode synthetic --snr-db -8 --host 127.0.0.1 --port 50000
```
(the simulator sends UDP; for a TCP smoke test, see
`02_Testing_Module/test_pipeline.py::test_ethernet_receiver_tcp_roundtrip`
for a minimal TCP server example you can adapt).
