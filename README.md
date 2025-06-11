# Crowd-Count: RF-based Crowd Counting using Synthetic and Real mmWave Data

This project implements a deep learning pipeline for crowd counting using RF-based range profiles. The model is trained using both synthetic data (generated from physical models) and real mmWave radar data from the People-Gait dataset.

The project follows a hybrid residual learning framework, where a simple physics-based peak detection algorithm is used to estimate an initial baseline count, and a Temporal Convolutional Network (TCN) is trained to predict the residual correction to this estimate.

---


