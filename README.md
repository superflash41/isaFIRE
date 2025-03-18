# Wildfire Classifier - 1INF52 Project

<p align="center">
    <img src="https://img.shields.io/badge/Python-f9e2af?logo=python&logoColor=black" alt="Python" />
    <img src="https://img.shields.io/badge/TensorFlow-f2cdcd?logo=tensorflow&logoColor=black" alt="TensorFlow" />
</p>

Repo with our Deep Learning approach for **Wildfire Detection** on the
[FLAME](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs)
dataset. We train three convolutional neural network (CNN) models (DenseNet, ResNet, Xception) via transfer learning,
then combine them in an ensemble. Our final model is intended to detect "Fire" vs. "No Fire" in UAV images.

## Pipeline

1. **Data Preprocessing**: resize and augment the FLAME dataset images.
2. **Model Training**: 
   - train each CNN individually with Keras Tuner-optimized hyperparameters.
   - hyperparameters (example from final run):
     - **Xception**: 25 unfrozen layers, 0.45 dropout, 0.001 L2, LR=0.00541  
     - **DenseNet**: 20 unfrozen layers, 0.35 dropout, 0.001 L2, LR=0.00147  
     - **ResNet**: 45 unfrozen layers, 0.40 dropout, 0.0005 L2, LR=0.00093  
3. **Ensemble**: merge each model’s prediction via simple averaging.
4. **Evaluation**: compute accuracy, F1-score, confusion matrix, and ROC-AUC on test set.

To run everything in one shot:
```bash
chmod +x ./scripts/run_pipeline.sh
./scripts/run_pipeline.sh
```

## Web App and Hugging Face

We created a minimal FastAPI [web application](https://github.com/superflash41/isaFIRE) where one can upload an image
and see if the model detects fire.

The trained models can also be found on [Hugging Face](https://huggingface.co/superflash41/fire-chad-detector-v1.0).

## Documentation

The project's final report with our research's explanation can be found in the [`report/`](report) folder
and the presentation and poster can be found on the [`presentation/`](presentation) folder. 

