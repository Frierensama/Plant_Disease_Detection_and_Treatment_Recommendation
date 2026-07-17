# Plant_Disease_Detection_and_Treatment_Recommendation

A GenAI-based web-application that detects plant diseases from leaf images using a Convolutional Neural Network (CNN) and generates treatment recommendations using a Hugging Face Large Language Model (LLM).

deployment
```bash
https://plant-disease-detection-and-treatment-recommendation-bleh.streamlit.app/
```

---

## Result

This project classifies plant diseases from leaf images and provides AI-generated information, including:

- Disease Overview
- Symptoms
- Causes
- Treatment Recommendations
- Prevention Tips

The application is built using **PyTorch**, **Streamlit**, and the **Hugging Face Inference API**.

---

## Required Libraries

- Python
- PyTorch
- TorchVision
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit
- Hugging Face Inference API

---

## Root Directory

```text
Plant_Disease_Detection_and_Treatment_Recommendation/
│
├── models/
│   └── model_v05.pth
│
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_dev_cnn.ipynb
│   ├── transfering_learning_RESNET.ipynb
│   ├── transfering_learning_ENET.ipynb
│   ├── transfering_learning_mbnetv2.ipynb
│   ├── Models_Evaluation.ipynb
│   └── Custom_scripts.py
│
├── pickle_items/
│
├── app.py
├── hugging_face_response.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Models Development

The project includes the following models:

- Custom CNN
- ResNet50 (Transfer Learning)
- EfficientNetB0 (Transfer Learning)
- MobileNetV2 (Transfer Learning)

For Deployment purposes im using **Custom CNN** since it uses less memory than pre-trained models at a cost of 0.5% accuracy.

---

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Per-Class Accuracy
- Misclassified Samples

---

## how it works

1. Upload a leaf image.
2. The CNN model predicts the disease.
3. The predicted disease is sent to a Hugging Face LLM.
4. The LLM generates:
   - Disease Overview
   - Symptoms
   - Causes
   - Treatment
   - Prevention Tips
5. The results are displayed in the Streamlit application.

---

## Kaggle Dataset

This project uses the **PlantVillage** dataset containing healthy and diseased plant leaf images across multiple crop species.

---

## License

Bleh.