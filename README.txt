NOTE: See README.md at the project root for the current, up-to-date
overview (both models trained, frontend built, real-dataset support,
verified results). This file is kept as an early quick-reference and is
otherwise still accurate for the module layout.

NPOL TONAL DETECTION SUITE
===========================

Robust Low-SNR Passive Sonar Tonal Detection using Transformer-Based
Spectrogram Segmentation (SegFormer-B0). Target: -7 dB to -10 dB SNR,
real-time, deployable on a standard laptop (CPU only).

START HERE:
    00_Documentation/README.md   - full architecture & module guide
    00_Documentation/SETUP.md    - installation & how to run every stage
    00_Documentation/RESULTS.md  - what has been run/measured so far,
                                    and what remains before production use

FASTEST START:
    pip install -r 12_Requirements/requirements.txt
    python main.py
    -> numbered menu: Train / Test / Live Detection / Signal Generator /
       Dataset Generator / Model Evaluation / Deployment / Ethernet /
       Settings. Every module reads its defaults from config.yaml.

QUICK START (module by module):
    pip install -r 12_Requirements/requirements.txt
    cd 02_Testing_Module   && pytest -v
    cd 01_Training_Module  && python train.py --model segformer
    cd 06_Model_Evaluation && python evaluate.py
    cd 03_Live_Detector    && python live_detector.py --source synthetic --snr-db -8
    cd ..                  && python deploy.py            # standalone, no training code

Folder structure matches Section 10 of Project_proposal.docx, plus
config.yaml / main.py / deploy.py / logs/ / reports/ added at the
project root -- see 00_Documentation/README.md Section 3 for the full,
current folder table.
