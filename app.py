import os
import logging

# Suppress TensorFlow logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3 = FATAL only
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import streamlit as st
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # Silence deprecation warnings

from tensorflow.keras import layers, models
import numpy as np
from PIL import Image

# Configuration
IMG_HEIGHT = 180
IMG_WIDTH = 180
BATCH_SIZE = 32
MODEL_PATH = 'tamil_kalvettu_model.keras'
CLASS_NAMES_PATH = 'class_names.npy'

def create_model(num_classes):
    # This is the "Old Very First Code" - Simple CNN Model
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def load_data(data_dir):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )
    class_names = train_ds.class_names
    return train_ds, val_ds, class_names

def predict_image(model, image, class_names):
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    return predicted_class, confidence

def main():
    st.set_page_config(page_title="Tamil Kalvettu OCR", page_icon="🗿", layout="wide")
    
    # Custom CSS for Premium Look
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #1e1e2f 0%, #2a2a40 100%);
            color: #ffffff;
        }
        .main .block-container {
            padding-top: 2rem;
        }
        h1, h2, h3 {
            font-family: 'Outfit', sans-serif;
            color: #ffffff;
            font-weight: 700;
        }
        .stButton>button {
            background: linear-gradient(90deg, #ff8a00, #e52e71);
            color: white;
            border: none;
            padding: 0.5rem 1.5rem;
            border-radius: 10px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(229, 46, 113, 0.4);
        }
        .stFileUploader {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px dashed #e52e71;
            border-radius: 10px;
            padding: 1rem;
        }
        .css-1d391kg {
            padding-top: 3.5rem;
        }
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #161623;
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🗿 Tamil Kalvettu Script Recognition")
    st.markdown("""
    <div style="background-color: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px; border-left: 5px solid #ff8a00;">
        <h4 style="margin:0; color: #ffca80;">Preserving Heritage with AI</h4>
        <p style="margin-top:10px; font-size: 1.1em; opacity: 0.9;">
            Upload an image of an old Tamil Kalvettu (inscription) to reveal its modern meaning. 
            Our advanced Neural Network analyzes the strokes and patterns to bridge the gap between history and today.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("## ⚙️ Control Panel")
    
    dataset_path = st.sidebar.text_input("Dataset Directory", value="dataset")
    
    train_btn = st.sidebar.button("🚀 Train New Model", help="Click to start training the neural network on your dataset.")
    
    if train_btn:
        if os.path.exists(dataset_path) and len(os.listdir(dataset_path)) > 0:
            with st.spinner('✨ Training neural network...'):
                progress_bar = st.sidebar.progress(0)
                try:
                    # 1. Load Data
                    train_ds, val_ds, class_names = load_data(dataset_path)
                    
                    # 2. Create Model
                    model = create_model(len(class_names))
                    
                    # 3. Train
                    history = model.fit(
                        train_ds,
                        validation_data=val_ds,
                        epochs=10 
                    )
                    progress_bar.progress(100)
                    
                    # 4. Save Model & Classes
                    model.save(MODEL_PATH)
                    np.save(CLASS_NAMES_PATH, class_names)
                    
                    st.sidebar.success("✅ Model trained & saved!")
                    st.sidebar.markdown(f"**Detected Classes:** `{', '.join(class_names)}`")
                    
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")
        else:
            st.sidebar.error("❌ Dataset not found.")

    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader("Drop your Kalvettu image here", type=["jpg", "png", "jpeg"])

    model = None
    class_names = []
    
    # Try to load model
    if os.path.exists(MODEL_PATH) and os.path.exists(CLASS_NAMES_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            class_names = np.load(CLASS_NAMES_PATH)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.warning("The previously saved model might be incompatible. Please Train New Model again.")
            model = None

    with col2:
        st.subheader("🤖 Analysis Results")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Source Image', use_column_width=True)
            
            if model is not None:
                if st.button("🔍 Decode Inscription"):
                    img_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
                    with st.spinner('Thinking...'):
                        predicted_class, confidence = predict_image(model, img_resized, class_names)
                    
                    st.markdown(f"""
                    <div style="text-align: center; padding: 20px; background: rgba(0, 255, 128, 0.1); border-radius: 15px; border: 2px solid #00ff80;">
                        <h2 style="color: #00ff80; margin: 0;">{predicted_class}</h2>
                        <p style="margin: 5px 0 0 0; opacity: 0.7;">Confidence: {confidence:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(int(confidence))
            else:
                st.info("👈 Please train the model first to see predictions.")
        else:
            st.write("Waiting for image upload...")

if __name__ == "__main__":
    main()
