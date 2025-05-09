# Wildfire Classifier - *1INF52* Project

<p align="center">
    <img src="https://img.shields.io/badge/Python-f9e2af?logo=python&logoColor=black" alt="Python" />
    <img src="https://img.shields.io/badge/TensorFlow-f2cdcd?logo=tensorflow&logoColor=black" alt="TensorFlow" />
</p>

## Project Overview

Deep learning has demonstrated exceptional performance in image analysis, particularly for detecting fire and smoke, outperforming many traditional techniques. This project introduces a deep learning model for real-time wildfire detection using drone-captured images, aimed at rapid and accurate identification of active fires.

This repository contains the code for our project, which is part of the course *1INF52 - Deep Learning* at [PUCP](https://www.pucp.edu.pe/).

We worked with the [FLAME](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs) dataset.

---

## Pipeline

We trained three CNNs via *transfer learning*, and combined their predictions in an *ensemble*.

1. *Data Preprocessing*: resized and used data augmentation on FLAME images.
2. *Model Training*: 
   - trained each CNN individually with *Keras Tuner*-optimized hyperparameters
   - hyperparameters:
     - **Xception**: $25$ unfrozen layers, $0.45$ dropout, $0.001$ L2, LR= $0.00541$  
     - **DenseNet**: $20$ unfrozen layers, $0.35$ dropout, $0.001$ L2, LR= $0.00147$
     - **ResNet**: $45$ unfrozen layers, $0.40$ dropout, $0.0005$ L2, LR= $0.00093$  
3. *Ensemble*: merged predictions via simple averaging
4. *Evaluation*: computed accuracy, F1-score, confusion matrices, and ROC-AUC

To run the pipeline we recommed having Python $3.8$+:
```bash
chmod +x ./scripts/run.sh
./scripts/run.sh
```

---

## Final Models

We created a minimal FastAPI [web application](https://github.com/superflash41/isaFIRE-demo-app) where one can test the models by uploading images of wildfire.

Trained models are on [Hugging Face](https://huggingface.co/superflash41/fire-chad-detector-v1.0).

## Documentation

The project's final report with our research's explanation can be found in the [`report/`](report/) folder. Slides from the presentation and poster can be found on the [`presentation/`](presentation/) folder. 

