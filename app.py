import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from notebooks.Custom_scripts import CustomNN
import torch
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from hugging_face_response import gen_response
import cv2

load_dotenv()

st.set_page_config(page_title='Plant Disease Detection and Treatment Reccomendation 🍀', page_icon='☘️', layout='wide')

HF_TOKEN = os.getenv('hf_token')
# cnn model load
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cnn = CustomNN(38)
cnn.load_state_dict(torch.load('models/model_v05.pth',map_location=device))
cnn.eval()

menu = st.sidebar.radio(
    'Navigation',
    [
     'Home',
     'Model Comparision',
     'Predict/Treatment'
     ]
)

image_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])


print(os.getcwd())
with open('pickle_items/class_names.pkl','rb') as f:
    class_names = pickle.load(f)

# dataframes
with open('pickle_items/models_eval_df.pkl','rb') as f:
    model_eval_df = pickle.load(f)
with open('pickle_items/misclassified_df.pkl','rb') as f:
    misclassified_df = pickle.load(f)
with open('pickle_items/df_per_class_accuracy.pkl','rb') as f:
    per_class_accuracy_df = pickle.load(f)

# cnn losses and accuracies
with open('pickle_items/cnn/train_losses_cnn.pkl','rb') as f:
    train_losses_cnn = pickle.load(f)
with open('pickle_items/cnn/train_accs_cnn.pkl','rb') as f:
    train_accs_cnn = pickle.load(f)
with open('pickle_items/cnn/valid_accs_cnn.pkl','rb') as f:
    valid_accs_cnn = pickle.load(f)
with open('pickle_items/cnn/valid_losses_cnn.pkl','rb') as f:
    valid_losses_cnn = pickle.load(f)


num_features = len(class_names)

if menu == 'Home':
    st.title('**AI-Powered Plant Disease Detection and Treatment Recommendation using Deep Learning and Hugging Face Generative AI.**')

    st.write(''' 
        *This application is built using PyTorch. 
        I have used few pre-trained models too to compare the performance.
        For deployment purposes, i've used CustomCNN (which i have created) over pre-trained models.*
        ''')
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric('**Accuracy**','0.96')
    col2.metric('**Precision**','0.97')
    col3.metric('**Recall**','0.97')
    col4.metric('**F1 Score**','0.96')


    d1, d2 = st.columns(2)

    fig1 = plt.figure(figsize=(10,5))
    plt.plot(range(1,31),train_accs_cnn, label='Train')
    plt.plot(range(1,31),valid_accs_cnn, label='Valid')
    plt.title('Increase in Accuracy per Epoch - CNN')
    plt.yticks(np.arange(0.5, 1.1, 0.1))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    d1.pyplot(fig1)
    fig2=plt.figure(figsize=(10,5))
    plt.plot(range(1,31),train_losses_cnn, label='Train')
    plt.plot(range(1,31),valid_losses_cnn, label='Valid')
    plt.title('Loss per Epoch - CNN')
    plt.yticks(np.arange(0.5, 2.1, 0.5))
    plt.xlabel('Epochs')
    plt.ylabel('Loss per Batch')

    plt.legend()
    d2.pyplot(fig2)

    st.header('Traning and Validation')
    st.write('*The model is trained for 4 hours on google collab gpu for 30 epochs.')
    f1,f2,f3 = st.columns(3)
    f1.metric('Train Size','70295')
    f2.metric('Validation Size','17572')
    f3.metric('Accuracy on Test (33 labels)', '100%')

    st.header('Per Class Accuracy')
    st.dataframe(per_class_accuracy_df)

    st.write('*I have trained pre-trained models with default classification layout (ie. one Linear Layer -- input_features to num_classes). Resnet50 out-performed all models after 10 epochs. for deployment purposes im using my model.*')

elif menu=='Model Comparision':
    st.title('Model Performance Comparison ')
    st.write('*I have trained pre-trained models with default classification layout (ie. one Linear Layer -- input_features to num_classes). **Resnet50** out-performed all models after 10 epochs. for deployment purposes im using my model.*')
    st.dataframe(model_eval_df.sort_values('Accuracy',ascending=False))
    st.divider()

    st.subheader("Custom CNN Architecture")
    st.markdown("""
    ```
    Input Image (3 × 224 × 224)
            │
            ▼
    ──────────────────────────────────────
    Feature Extraction
    ──────────────────────────────────────
    Conv2D (3 → 16, Kernel=3×3, Padding='same')
    ReLU
    BatchNorm2D
    MaxPool2D (2×2)

    Conv2D (16 → 32, Kernel=3×3, Padding='same')
    ReLU
    BatchNorm2D
    MaxPool2D (2×2)

    Conv2D (32 → 64, Kernel=3×3, Padding='same')
    ReLU
    BatchNorm2D
    MaxPool2D (2×2)

    Dropout (0.2)

    ──────────────────────────────────────
    Classifier
    ──────────────────────────────────────
    Flatten

    Linear (50176 → 128)
    BatchNorm1D
    ReLU
    Dropout (0.3)

    Linear (128 → 64)
    BatchNorm1D
    ReLU
    Dropout (0.4)

    Linear (64 → 38)

    ──────────────────────────────────────
    Output
    ──────────────────────────────────────
    38 Plant Disease Classes
    Softmax (during inference)
    ```
    """)

    st.divider()
    st.header('Mis-Classified Labels ~5-6% pr model')
    st.dataframe(misclassified_df)

elif menu =='Predict/Treatment':
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    
    selected_image = st.selectbox('Choose an image', os.listdir(os.getcwd()+'/test'))

    if uploaded_file is not None:

        # Image processing to (1,3,224,224)
        image = Image.open(uploaded_file)
        img = image_transform(image)
        img = img.unsqueeze(0)

        # cnn prediction
        output = cnn(img)
        _,pred_label = torch.max(output,axis=1)
        pred = class_names[pred_label]
        c0,c1 , c2 = st.columns(3)

        c0.subheader(f'**Prediction Confidence - {per_class_accuracy_df['Accuracy_CNN'][int(pred_label)]:.3f}**')
        c1.image(uploaded_file)
        c2.subheader(f'**Plant Class - {pred}**')
        st.divider()
        if pred.split('___')[1] != 'healthy':
            st.header('Treatment/Suggestion')
            if st.button('Need Treatment Reccomendations?'):
                response = gen_response(pred, hf_token=HF_TOKEN)
                st.write(f'*{response}*')
        else:
            st.header('No disease detected. Healthy plant.')
            st.write('*I aint gonna waste tokens*')

    if selected_image is not None and uploaded_file is None:
            uploaded_file = os.getcwd()+'/test/'+selected_image
            # Image processing to (1,3,224,224)
            image = Image.open(uploaded_file)
            img = image_transform(image)
            img = img.unsqueeze(0)
    
            # cnn prediction
            output = cnn(img)
            _,pred_label = torch.max(output,axis=1)
            pred = class_names[pred_label]
            c0,c1 , c2 = st.columns(3)
    
            c0.subheader(f'**Prediction Confidence - {per_class_accuracy_df['Accuracy_CNN'][int(pred_label)]:.3f}**')
            c1.image(uploaded_file)
            c2.subheader(f'**Plant Class - {pred}**')
            st.divider()
            if pred.split('___')[1] != 'healthy':
                st.header('Treatment/Suggestion')
                if st.button('Need Treatment Reccomendations?'):
                    response = gen_response(pred, hf_token=HF_TOKEN)
                    st.write(f'*{response}*')
            else:
                st.header('No disease detected. Healthy plant.')
                st.write('*I aint gonna waste tokens*')