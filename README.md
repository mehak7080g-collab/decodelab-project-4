# Project 4: Image Recognition (Basic)

## Overview

This project implements a basic image recognition workflow using a **pre-trained ResNet18 model** from TorchVision.

The goal is to demonstrate how an existing AI model can be integrated into a functional Python workflow to recognize the contents of a sample image and clearly display the model's predictions.

## Project Requirements

The project follows the provided Project 4 requirements:

- Use a pre-trained model or simple AI library
- Perform recognition on a sample image
- Display the output clearly

## Technology

- Python
- PyTorch
- TorchVision
- Pillow
- Pre-trained ResNet18
- Image classification

## How It Works

```text
Input Image
     |
     v
Image Preprocessing
     |
     v
Pre-trained ResNet18
     |
     v
Model Predictions
     |
     v
Top-K Results + Confidence
```

## Installation

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

Provide the path to an image:

```bash
python main.py sample.jpg
```

To display a different number of predictions:

```bash
python main.py sample.jpg --top-k 3
```

The first run may download the pre-trained ResNet18 weights automatically.

## Example Output

```text
============================================================
        PROJECT 4: IMAGE RECOGNITION (BASIC)
============================================================
Input image: sample.jpg

Top predictions:
1. golden retriever — 82.31%
2. Labrador retriever — 9.47%
3. tennis ball — 2.15%

Recognition completed successfully.
```

*The exact predictions and confidence values depend on the input image.*

## Project Structure

```text
project4-image-recognition/
├── main.py
├── requirements.txt
└── README.md
```

## Learning Outcomes

This project demonstrates:

- Using a pre-trained AI model
- Image preprocessing
- Model inference
- Reading model outputs
- Ranking predictions
- Displaying confidence scores
- Integrating an AI library into a Python application

## Scope

This is a basic image-classification implementation. It uses a pre-trained model rather than training a new model from scratch.

## Future Improvements

- Add a graphical user interface
- Add webcam recognition
- Support batch image classification
- Add image upload through a web application
- Add object detection
- Store recognition history
- Integrate the model into an API
