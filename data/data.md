# FLAME Dataset

The **FLAME (Fire Luminosity Airborne-based Machine learning Evaluation) Dataset** is a comprehensive collection of aerial imagery captured using drones during prescribed pile burns in Northern Arizona, USA. This dataset is designed to aid in the development and evaluation of machine learning models for wildfire detection and segmentation.

## Dataset Overview

- **Total Images:** Approximately 47,992 labeled frames
  - **Training Set:** 39,375 frames
  - **Test Set:** 8,617 frames
- **Categories:**
  - **Fire:** Frames containing visible fire
  - **No Fire:** Frames without visible fire
- **Annotations:**
  - **Classification Labels:** Binary labels indicating the presence or absence of fire
  - **Segmentation Masks:** Pixel-wise annotations for precise fire boundary detection (2,003 masks available)

## Data Collection

The dataset includes:

- **Raw Videos:** Captured using Zenmuse X4S camera during prescribed burns
- **Thermal Footage:** Recorded by an infrared thermal camera
- **Labeled Frames:** Extracted from videos and annotated for classification and segmentation tasks

## Accessing the Dataset

The FLAME Dataset is publicly available on IEEE DataPort:

[FLAME Dataset on IEEE DataPort](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs)

## Citation

If you use the FLAME Dataset in your research, please cite the following publication:

```bibtex
@article{Shamsoshoara2021,
  title={Aerial Imagery Pile Burn Detection Using Deep Learning: The FLAME Dataset},
  author={Shamsoshoara, Alireza and Afghah, Fatemeh and Razi, Abolfazl and Zheng, Liming and Fulé, Peter and Blasch, Erik},
  journal={Computer Communications},
  year={2021},
  publisher={Elsevier}
}
