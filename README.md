
# GOYAMA: Rice Disease Identification Using Deep Learning

GOYAMA is a deep learning model built with PyTorch that identifies diseases in rice leaves from photos. The model can accurately detect four types of rice diseases: Bacterial blight, Blast, Brown spot, and Tungro.



## Dataset

The model is trained using the [Rice Leaf Dataset from Mendeley Data](https://www.kaggle.com/datasets/maimunulkjisan/rice-leaf-dataset-from-mendeley-data) available on Kaggle.

## Diseases Identified

The model can identify the following rice diseases:
- **Bacterial blight**: A serious disease caused by the bacterium *Xanthomonas oryzae* which leads to wilting of seedlings and yellowing and drying of leaves.
  
  <p align="center">
  <img src="https://github.com/jaliyanimanthako/Goyama/assets/161110418/468ec9eb-a1ab-453f-b6e2-5e345219a0f4" >
</p>
  
  
- **Blast**: Caused by the fungus *Magnaporthe oryzae*, this disease leads to lesions on leaves, collars, nodes, and panicles'
  
  <p align="center">
  <img src="https://github.com/jaliyanimanthako/Goyama/assets/161110418/98298f46-d167-40ef-a175-98e4b0b0692a" >
</p>
  
- **Brown spot**: A fungal disease caused by *Cochliobolus miyabeanus* that results in brown lesions on the leaves and grains, reducing the grain quality and yield.
    <p align="center">
  <img src="https://github.com/jaliyanimanthako/Goyama/assets/161110418/1c5aa2eb-28be-4992-be3e-843f0ebec259" >
</p>

- **Tungro**: Caused by a complex of two viruses, the Rice tungro bacilliform virus (RTBV) and Rice tungro spherical virus (RTSV), leading to stunted growth, yellow-orange discoloration, and reduced tillering.
      <p align="center">
  <img src="https://github.com/jaliyanimanthako/Goyama/assets/161110418/c6b2dec7-85dd-49b2-a620-4caf71f13f5b" >
</p>

  
### Prerequisites

- Python 3.7+
- PyTorch
- OpenCV
- Albumentations
- tqdm
## Data Augmentation

Data augmentation is performed to increase the diversity of the training dataset and improve the model's robustness.

### Augmentation Pipeline

The augmentation pipeline includes:
- Rotation
- ShiftScaleRotate
- Horizontal and Vertical Flips
- Random Brightness and Contrast
- Hue, Saturation, and Value shifts
- Gaussian Noise
- Sharpening


## Model Architecture

Consists of several convolutional layers, followed by fully connected layers.
![Architecture](https://github.com/jaliyanimanthako/Goyama/assets/161110418/e6c4a630-0590-4040-b9a0-057fe871479d)


### Model Layers

- **Conv Layers**: 6 convolutional layers with increasing filter sizes.
- **Pooling Layers**: Max pooling after each convolutional block.
- **Fully Connected Layers**: A sequence of fully connected layers with Batch Normalization and ReLU activation.
- **Dropout**: Applied before the fully connected layers to prevent overfitting.

## Evaluation

After training, the model can be evaluated on a test set to measure its performance. Both the test and validation accuracy end at around 98%.
![Graph](https://github.com/jaliyanimanthako/Goyama/assets/161110418/1cf6ce52-bcef-46e1-a6c9-028b669281ff)


## Results

The model achieves high accuracy in identifying the four rice diseases from the leaf images. 
![Prediction](https://github.com/jaliyanimanthako/Goyama/assets/161110418/eb131fee-ff43-4875-a25e-f7f14b741c6d)


## Contribution

Feel free to fork this repository, make your improvements, and create pull requests. Contributions are welcome!

## Acknowledgements

- The authors of the Rice Leaf Dataset from Mendeley Data.
- The contributors of PyTorch, OpenCV, and Albumentations libraries.

## Developed With

- **PyTorch**: An open source machine learning framework that accelerates the path from research prototyping to production deployment.
    <p align="center">
  <img src="https://github.com/jaliyanimanthako/Goyama/assets/161110418/06a3c4e6-da40-4587-8646-e62e02226a4e" >
</p>
  
## Citations

This project references the following research papers:

1. [Transfer Learning for Multi-Crop Leaf Disease Image Classification using Convolutional Neural Network VGG](https://www.sciencedirect.com/science/article/pii/S2589721721000416#bb0035)
2. [Deep Learning-Based Leaf Disease Detection in Crops Using Images for Agricultural Applications](https://doi.org/10.3390/agronomy12102395)
3. [Convolutional Neural Network for Automatic Identification of Plant Diseases with Limited Data](https://doi.org/10.3390/plants10010028)

