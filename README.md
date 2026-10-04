# Multispectral Voice Extraction Biometric System for Wearable Medical Devices

## Overview

This work proposes **EFSPEN (Efficient Full-band and Sub-band Path Extension Network)**, a lightweight speech-enhancement front-end designed to improve noisy speech before **speaker verification**. EFSPEN extends the FSPEN backbone with a compact **frequency-wise multi-head self-attention** module while keeping the model suitable for resource-constrained and wearable devices.

<img width="2243" height="1120" alt="diagram-export-4-6-2026-10_08_47-AM_modif_2" src="https://github.com/user-attachments/assets/b0cad2f6-d27c-47c9-b29b-1ade58ca5c17" />

## Main Contribution

- Adds a **4-head Frequency Attention** module to FSPEN.
- Processes both **global full-band** and **local sub-band** spectral features.
- Uses only **39,117 trainable parameters**.
- Designed as a **preprocessing front-end**, not as a replacement for the speaker-recognition model.

<img width="1568" height="830" alt="Figure 2_modif" src="https://github.com/user-attachments/assets/0e56578c-b4e7-4946-8378-9b8149677173" />

<img width="1321" height="256" alt="Biometric_Verification_Pipeline01 drawio" src="https://github.com/user-attachments/assets/fcba3f6a-e139-493d-8831-483695bb1083" />

## Experimental Setup

- **Speech enhancement training:** VoiceBank+DEMAND, 16 kHz
- **Training samples:** ~11.6K
- **Test samples:** 824
- **Input duration:** up to 4 s
- **STFT:** 512-point FFT, 256-sample hop, 257 frequency bins
- **Speaker embedding model:** SpeechBrain **ECAPA-TDNN**
- **Biometric metric:** Equal Error Rate (**EER**)

## Key Results

| Evaluation | Result |
|---|---:|
| VoxCeleb1: ECAPA-TDNN | 6.87% EER |
| VoxCeleb1: EFSPEN + ECAPA-TDNN | **5.00% EER** |
| VoxCeleb1: EFSPEN + adapted ECAPA-TDNN | **3.00% EER** |

The adapted ECAPA-TDNN was trained through **knowledge distillation** so that embeddings extracted from EFSPEN-enhanced speech better match those produced from the original speech.

## Statistical Result

On VoxCeleb1:

- **EFSPEN + unadapted ECAPA-TDNN:** 5.00% EER, 95% CI: 2.51–7.04%
- **EFSPEN + adapted ECAPA-TDNN:** 3.00% EER, 95% CI: 1.72–5.16%
- Paired difference: **−2.00 percentage points**, 95% CI: −4.20 to 0.70 pp

The point estimate improves after adaptation, but the confidence interval includes zero, so the difference is **not statistically significant at α = 0.05** on the evaluated subset.

## Takeaway

EFSPEN shows that a very compact speech-enhancement network can improve perceptual speech quality and support more robust speaker verification in noisy conditions. The strongest VoxCeleb1 result is obtained when the downstream ECAPA-TDNN verifier is adapted to EFSPEN-enhanced speech.

## Future Work

A planned extension is a **hybrid quantum-classical EFSPEN**, integrating parameterized quantum circuits using PennyLane.
