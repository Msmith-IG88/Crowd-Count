# Crowd-Count: RF-based Crowd Counting using Synthetic and Real mmWave Data

This project implements a  deep learning pipeline for crowd counting using RF-based sensing. The model is trained primarily on synthetic data generated from physics motion models and later tuned using real-world mmWave radar data collected from the [People-Gait dataset](https://github.com/mmGait/people-gait).

The project follows a residual learning framework, where:

- A simple peak detection algorithm estimates an initial baseline count.
- A Temporal Convolutional Network (TCN) is trained to predict a residual correction to this baseline.
- The final predicted count is computed as:  
  `predicted_count = baseline_estimate + residual_correction`

The neural network architecture, data generation framework, and training pipeline are inspired by these papers:

- Pallaprolu, Hurst, and Mostofi, *Crowd Analytics with a Single mmWave Radar*, MobiCom 2024. [[PDF]](https://web.ece.ucsb.edu/~ymostofi/papers/PallaproluHurstMostofi_MobiCom24.pdf)
- Korany et al., *Teaching RF to Sense without RF Training Measurements*, Ubicomp 2020. [[PDF]](https://web.ece.ucsb.edu/~ymostofi/papers/Ubicom20_KoranyKaranamCaiMostofi.pdf)
- People-Gait Dataset (used for real-world evaluation): [People-Gait GitHub](https://github.com/mmGait/people-gait)

## Repository Overview

- `synthetic_data.py` — Synthetic data generator using range-only profiles with simplified occlusion modeling and Gaussian spreading.
- `real_sequence_dataset.py` — Loads and preprocesses the real People-Gait mmWave dataset into range profiles.
- `model.py` — Residual learning Temporal Convolutional Network (CrowdTCN) model.
- `train.py` — Training loop using synthetic data.
- `tune.py` — Fine-tuning using small amounts of real data.
- `real_eval.py` — Evaluation script on real-world data with full metrics.
- `graph.py` — Visualization utilities to compare synthetic vs real data heatmaps and profiles.

## Dataset

- Synthetic data is generated entirely using a physics base model of multiple people walking randomly or periodically in a room.
- Real-world data comes from the TI 77 GHz mmWave radar used in the People-Gait dataset.

## Results Summary

- The model trained purely on synthetic data achieved an MAE of ~1.02 on real data.
- After fine-tuning with a small number of real samples, MAE improved to ~0.32 and accuracy increased from 44% to 69%.

## References

[1] Pallaprolu, Hurst, Mostofi. *Crowd Analytics with a Single mmWave Radar*, ACM MobiCom 2024.  
[2] Korany, Karanam, Cai, Mostofi. *Teaching RF to Sense without RF Training Measurements*, Ubicomp 2020.  
[3] Wu et al., *Gait Recognition for Co-Existing Multiple People Using Millimeter Wave Sensing*, IEEE Transactions on Mobile Computing, 2021.  
[4] People-Gait Dataset: https://github.com/mmGait/people-gait

---

**Author:** Michael Smith (ECE 250 Final Project)




