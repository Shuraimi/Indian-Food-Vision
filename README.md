# Indian-Food-Vision

This a Deep Learning project which uses the Khana dataset to perform image classification. I'm doing this project as an inspiration from the nutrify app by Daniel Bourke Sir. I really liked his project so I wanted to try and create an app similar to his but on Indian foods .

So I've chosen the Khana dataset from the paper https://arxiv.org/abs/2509.06006 which has 80 categories of Indian foods and almost 1,30,000 images in total.

Complete Guide: Building an Indian Food Object Detection App
Overview
This guide outlines a strategic, step-by-step process for building an Indian food object detection app using a classification-only dataset like Khana. It avoids the tedious task of manually labeling 130,000 images by using a "human-in-the-loop" approach with AI-assisted labeling and active learning.
Part 1: Train the image classification model
This step leverages the existing class labels of the Khana dataset to create the foundational model.
Step 1.1: Set up the development environment and download the dataset
Create a Google Colab notebook . This provides a free, cloud-based Python environment with GPU access, which is essential for training deep learning models.
Download and organize the Khana dataset . Write a Python script to download the dataset and organize it into train, val, and testsubdirectories. Each subdirectory should contain folders for each food category, as required by standard image classification frameworks.
Step 1.2: Train the image classifier with transfer learning
Use a powerful pre-trained model . Instead of training from scratch, use a pre-trained model (eg, ResNet, EfficientNet, ViT) and fine-tune it for your specific 80 Indian food classes. This technique significantly speeds up development and improves accuracy.
Follow a comprehensive video tutorial on training an image classifier.
Video: Build an Image Classification Model with PyTorch
Key takeaway : Focus on achieving high classification accuracy. A well-performing classifier is the foundation for the next step.
Part 2: Generate pseudo-labels with AI assistance
This part focuses on automatically generating initial bounding box annotations, minimizing the need for manual work.
Step 2.1: Implement Grad-CAM for localization
Generate heatmaps : Write a script to implement Grad-CAM, which creates a visual heatmap showing which parts of an image your classification model focused on to make its prediction.
Automate bounding box creation : Extend the script to use the Grad-CAM heatmap to generate a rough bounding box around the most salient region.
Resource: Generating Bounding Boxes with Grad-CAM - A Kaggle notebook with code and visualization examples.
Step 2.2: Use Grounding DINO for precision (optional but highly recommended)
Install Grounding DINO : Use a zero-shot object detection model that can generate highly accurate bounding boxes based on a text prompt, without prior training.
Automate labeling with prompts : For each image, use your classification model's prediction as the text prompt for Grounding DINO. This will generate precise bounding box annotations automatically.
Resource: Zero-Shot Object Detection with Grounding DINO Colab - A notebook demonstrating the setup and usage.
Part 3: Refine labels and train the final model with active learning
This phase focuses your manual effort on the most challenging examples, allowing you to achieve a high-performance model with minimal human intervention.
Step 3.1: Set up an active learning loop
Integrate a labeling tool : Upload your dataset with the auto-generated pseudo-labels to an AI-assisted annotation tool like Roboflow or Encord.
Initial training : Train a preliminary object detection model (eg, YOLOv8) on a small subset of your most confident pseudo-labels.
Use the model to find uncertain images : The tool will use the model to identify images where it is least confident. These are the images you will prioritize for manual review.
Human-in-the-loop review : Manually review and correct the annotations only on the targeted subset of uncertain images.
Step 3.2: Iterate and train the final model
Retrain the model : Incorporate the newly corrected, high-quality data back into your training set.
Repeat : Continue the active learning loop by finding the next batch of uncertain images to label.
Train the final object detection model .
