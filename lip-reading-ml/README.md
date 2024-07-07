# Lip Reading Project

This project involves using a camera to read lips and interpret speech using machine learning models. The system captures video input from a webcam, processes the images to detect mouth movements, and uses a trained neural network to predict spoken words.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
  - [macOS](#macos)
  - [Windows](#windows)
- [Usage](#usage)
- [Requirements](#requirements)
- [Acknowledgements](#acknowledgements)

## Introduction

The Lip Reading project utilizes OpenCV for video processing, dlib for facial landmark detection, and a trained LipNet model for lip reading. This project demonstrates how machine learning can be applied to interpret speech from visual information.

## Installation

### macOS

Follow these steps to set up the project on macOS:

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the necessary model files:**
   - Download the `shape_predictor_68_face_landmarks.dat` file from [dlib's model zoo](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
   - Download the trained LipNet model `trained_sequence_model.h5` from your specified source.

5. **Place the model files:**
   - Move `shape_predictor_68_face_landmarks.dat` to the project directory.
   - Move `trained_sequence_model.h5` to the `models` directory.

### Windows

Follow these steps to set up the project on Windows:

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the necessary model files:**
   - Download the `shape_predictor_68_face_landmarks.dat` file from [dlib's model zoo](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
   - Download the trained LipNet model `trained_sequence_model.h5` from your specified source.

5. **Place the model files:**
   - Move `shape_predictor_68_face_landmarks.dat` to the project directory.
   - Move `trained_sequence_model.h5` to the `models` directory.

## Usage

To run the Lip Reading project, execute the following command:

```bash
python main.py
```

This script captures video from your webcam, processes the video frames to detect and analyze lip movements, and displays the predicted text on the screen.

## Requirements

The project requires the following packages:

- `opencv-python`
- `dlib`
- `numpy`
- `keras`
- `tensorflow`

You can install these packages using the `requirements.txt` file provided in the repository:

```bash
pip install -r requirements.txt
```

## Acknowledgements

- The `dlib` library for facial landmark detection.
- The creators of the LipNet model for their work on lip reading technology.
