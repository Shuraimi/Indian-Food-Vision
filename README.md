# Indian-Food-Vision

This a Deep Learning project which uses the Khana dataset to perform image classification. I'm doing this project as an inspiration from the nutrify app by Daniel Bourke Sir. I really liked his project so I wanted to try and create an app similar to his but on Indian foods .

So I've chosen the Khana dataset from the paper https://arxiv.org/abs/2509.06006 which has 80 categories of Indian foods and almost 1,30,000 images in total.



# Complete Guide: Building an Indian Food Object Detection App

## Overview
This guide outlines a strategic, step-by-step process for building an Indian food object detection app using a classification-only dataset like Khana. It avoids the tedious task of manually labeling 130,000 images by using a "human-in-the-loop" approach with AI-assisted labeling and active learning.

---

## Part 1: Train the image classification model

This step leverages the existing class labels of the Khana dataset to create the foundational model.

### Step 1.1: Set up the development environment and download the dataset
1. **Create a Google Colab notebook.**  
   This provides a free, cloud-based Python environment with GPU access, which is essential for training deep learning models.

2. **Download and organize the Khana dataset.**  
   Write a Python script to download the dataset and organize it into `train`, `val`, and `test` subdirectories. Each subdirectory should contain class-wise folders (e.g., each folder representing an Indian food class).

---

### Step 1.2: Train the image classifier with transfer learning
1. **Use a powerful pre-trained model.**  
   Instead of training from scratch, use a pre-trained model (e.g., ResNet, EfficientNet, ViT) and fine-tune it for your specific 80 Indian food classes.  
   This technique significantly speeds up development and improves accuracy.

2. **Follow a comprehensive video tutorial** on training an image classifier.  
   - **Video:** Build an Image Classification Model with PyTorch  
   - **Key takeaway:** Focus on achieving high classification accuracy. A well-performing classifier is the foundation for the next step.

---

## Part 2: Generate pseudo-labels with AI assistance

This part focuses on automatically generating initial bounding box annotations, minimizing the need for manual work.

### Step 2.1: Implement Grad-CAM for localization
1. **Generate heatmaps:**  
   Write a script to implement Grad-CAM, which creates a visual heatmap showing which parts of an image your classification model focused on to make its prediction.

2. **Automate bounding box creation:**  
   Extend the script to use the Grad-CAM heatmap to generate a rough bounding box around the most salient region.  
   - **Resource:** *Generating Bounding Boxes with Grad-CAM* — A Kaggle notebook with code and visualization examples.

---

### Step 2.2: Use Grounding DINO for precision (optional but highly recommended)
1. **Install Grounding DINO:**  
   Use a zero-shot object detection model that can generate highly accurate bounding boxes based on a text prompt, without prior training.

2. **Automate with prompts:**  
   For each image, use your classification model's predictions as input for Grounding DINO. This will generate refined bounding box annotations automatically.  
   - **Resource:** *Zero-Shot Object Detection with Grounding DINO Colab* — A notebook demonstrating the setup and usage.

---

## Part 3: Refine labels and train the final model with active learning

This phase focuses your manual effort on the most challenging examples, allowing you to achieve a high-performance model with minimal human intervention.

### Step 3.1: Set up an active learning loop
1. **Integrate a labeling tool:**  
   Upload your dataset with the auto-generated pseudo-labels to an AI-assisted annotation tool like Roboflow or Encord.

2. **Initial training:**  
   Train a preliminary object detection model (e.g., YOLOv8) on a small subset of your most confident pseudo-labels.

3. **Use the model to find uncertain images:**  
   The tool will use the model to identify images where it is least confident. These are the images you will prioritize for manual review.

4. **Human-in-the-loop review:**  
   Manually review and correct the annotations only on the targeted subset of uncertain images.

---

### Step 3.2: Iterate and train the final model
1. **Retrain the model:**  
   Incorporate the newly corrected, high-quality data back into your training set.

2. **Repeat:**  
   Continue the active learning loop by finding the next batch of uncertain images to label.

3. **Train the final object detection model.**  
   - **Video:** *YOLOv8: How to Train for Object Detection on a Custom Dataset* — A guide on training YOLOv8.

---

## Part 4: Deploy the model to an Android app

The final stage involves converting your trained model for mobile use and integrating it into an Android application.

### Step 4.1: Convert your model to TensorFlow Lite
1. **Export your YOLO model:**  
   The YOLOv8 framework allows for simple export to the `.tflite` format, optimized for on-device inference.  
   - **Resource:** *Export YOLOv8 to TFLite* — Documentation on the export process.

---

### Step 4.2: Build the Android application
1. **Follow the TensorFlow Codelab:**  
   Use this official Google resource to build an Android app that uses a custom TFLite object detection model.  
   - **Resource:** *Build and deploy a custom object detection model with TensorFlow Lite (Android)*

2. **Integrate your custom model:**  
   Follow the codelab's instructions to add your new `model.tflite` file and modify the app's code to run your custom model.

2. Download and organize the Khana dataset.
Write a Python script to download the dataset and organize it into train, val, and test subdirectories. Each subdirectory should contain class-wise folders (e.g., each folder representing an Indian food class).




---

Step 1.2: Train the image classifier with transfer learning

1. Use a powerful pre-trained model.
Instead of training from scratch, use a pre-trained model (e.g., ResNet, EfficientNet, ViT) and fine-tune it for your specific 80 Indian food classes.
This technique significantly speeds up development and improves accuracy.


2. Follow a comprehensive video tutorial on training an image classifier.

Video: Build an Image Classification Model with PyTorch

Key takeaway: Focus on achieving high classification accuracy. A well-performing classifier is the foundation for the next step.





---

Part 2: Generate pseudo-labels with AI assistance

This part focuses on automatically generating initial bounding box annotations, minimizing the need for manual work.

Step 2.1: Implement Grad-CAM for localization

1. Generate heatmaps:
Write a script to implement Grad-CAM, which creates a visual heatmap showing which parts of an image your classification model focused on to make its prediction.


2. Automate bounding box creation:
Extend the script to use the Grad-CAM heatmap to generate a rough bounding box around the most salient region.

Resource: Generating Bounding Boxes with Grad-CAM — A Kaggle notebook with code and visualization examples.





---

Step 2.2: Use Grounding DINO for precision (optional but highly recommended)

1. Install Grounding DINO:
Use a zero-shot object detection model that can generate highly accurate bounding boxes based on a text prompt, without prior training.


2. Automate with prompts:
For each image, use your classification model's predictions as input for Grounding DINO. This will generate refined bounding box annotations automatically.

Resource: Zero-Shot Object Detection with Grounding DINO Colab — A notebook demonstrating the setup and usage.





---

Part 3: Refine labels and train the final model with active learning

This phase focuses your manual effort on the most challenging examples, allowing you to achieve a high-performance model with minimal human intervention.

Step 3.1: Set up an active learning loop

1. Integrate a labeling tool:
Upload your dataset with the auto-generated pseudo-labels to an AI-assisted annotation tool like Roboflow or Encord.


2. Initial training:
Train a preliminary object detection model (e.g., YOLOv8) on a small subset of your most confident pseudo-labels.


3. Use the model to find uncertain images:
The tool will use the model to identify images where it is least confident. These are the images you will prioritize for manual review.


4. Human-in-the-loop review:
Manually review and correct the annotations only on the targeted subset of uncertain images.




---

Step 3.2: Iterate and train the final model

1. Retrain the model:
Incorporate the newly corrected, high-quality data back into your training set.


2. Repeat:
Continue the active learning loop by finding the next batch of uncertain images to label.


3. Train the final object detection model.

Video: YOLOv8: How to Train for Object Detection on a Custom Dataset — A guide on training YOLOv8.





---

Part 4: Deploy the model to an Android app

The final stage involves converting your trained model for mobile use and integrating it into an Android application.

Step 4.1: Convert your model to TensorFlow Lite

1. Export your YOLO model:
The YOLOv8 framework allows for simple export to the .tflite format, optimized for on-device inference.

Resource: Export YOLOv8 to TFLite — Documentation on the export process.





---

Step 4.2: Build the Android application

1. Follow the TensorFlow Codelab:
Use this official Google resource to build an Android app that uses a custom TFLite object detection model.

Resource: Build and deploy a custom object detection model with TensorFlow Lite (Android)



2. Integrate your custom model:
Follow the codelab's instructions to add your new model.tflite file and modify the app's code to run your custom model.




---

Would you like me to convert this Markdown into a formatted PDF or DOCX for easy reading and sharing?

 text prompt for Grounding DINO. This will generate precise bounding box annotations automatically.
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
