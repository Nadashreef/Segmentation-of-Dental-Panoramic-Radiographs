Dental Panoramic Radiograph Segmentation
Project Overview

This project focuses on the automatic segmentation of dental structures from panoramic radiographs. The goal is to assist dental professionals by providing accurate masks of teeth and surrounding structures, enabling further analysis, diagnosis, and treatment planning.

Dataset

The dataset used consists of panoramic radiographs and their corresponding masks. Each image has been carefully annotated to highlight relevant dental structures, facilitating supervised learning for segmentation tasks.

Features

Preprocessing of dental panoramic images for consistency and normalization.

Segmentation of teeth and other relevant dental structures.

Support for validation and training splits to evaluate model performance.

Visualization of original images alongside predicted masks for quality assessment.

Model

The project leverages deep learning techniques to perform image segmentation. The model architecture is designed to handle the challenges of dental panoramic images, including varying shapes, overlapping teeth, and image noise.

Results

Predictions are compared with ground truth masks, allowing evaluation of model accuracy and reliability. Visualization tools are included to display input images, masks, and model predictions side by side.

Directory Structure

src/: Contains the main source code for data preprocessing, transformation, and model training.

Notebook/: Contains exploratory notebooks and experiments related to the dataset and model.

logs/: Stores log files generated during training and experimentation.

data/ (optional): Contains images and masks if not using external storage.

Getting Started

To use this project:

Prepare your dataset of panoramic radiographs and corresponding masks.

Follow the preprocessing steps to normalize and structure the data.

Train the segmentation model with your data and evaluate the results.

Visualize predictions to ensure accuracy and reliability.
