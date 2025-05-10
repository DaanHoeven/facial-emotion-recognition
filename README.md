# Facial Emotion Recognition AI

This project is a Facial Emotion Recognition AI built with PyTorch. It can classify facial expressions into the following categories:

- Happy
- Angry
- Fear
- Neutral
- Sad
- Surprise

## 📂 Project Structure

```
├── models/                  # Trained models
├── snapshots/               # Latest best performing model (used by the AI)
├── images/
│   ├── train/
│   │   ├── angry/
│   │   ├── happy/
│   │   └── ...              # Training images for each emotion
│   └── validate/
│       ├── angry/
│       ├── happy/
│       └── ...              # Validation images for each emotion
├── example_images/          # Example images to test the AI
├── src/
│   └── model.ipynb          # Jupyter Notebook containing model training and usage
├── requirements.txt         # Python dependencies
└── README.md
```

## 📝 Installation

1. Clone this repository.
2. Install the required dependencies:

```bash
pip install -r .\requirements.txt
```

## 🚀 How to Use

Open the Jupyter notebook located at:

```
src/model.ipynb
```

At the bottom of the notebook, you will find a function:

```python
predict_emotions_from_image(image_path)
```

Use this function to test the AI on any image. Example images are available in the `example_images/` folder.

## 📊 Data

- **Training Data**: Located in `images/train/<emotion>/`
- **Validation Data**: Located in `images/validate/<emotion>/`

Each folder contains images labeled according to the emotion category.

## 📦 Models

- Trained models are stored in the `models/` folder.
- The best-performing and currently selected model for the AI is saved in the `snapshots/` folder.

## 📷 Example Images

Use images from the `example_images/` folder to quickly test the AI.

---

Enjoy experimenting with facial emotion recognition! 😄
