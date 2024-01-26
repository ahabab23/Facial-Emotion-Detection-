import os.path
import streamlit as st
import numpy as np
import base64
import cv2
import keras
import tensorflow as tf
import random
from PIL import Image,ImageOps
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Face Emotions Detection",
    page_icon=":angry:",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def loading_model():
  my_model=tf.keras.models.load_model("my_model")
  return my_model

model=loading_model()

# ###Background images
def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.

    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"

    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )
###sidebar
def sidebar_bg(side_bg):

   side_bg_ext = 'png'

   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )
def home():
    st.title("Home")
    st.write("Welcome to the Emotion Prediction App!")
    set_bg_hack("bagkground_image.jpg")

def app():
    st.title("App")

    set_bg_hack("background_img3.jpg")
    # side_bg = 'background_img2.jpg'
    # sidebar_bg(side_bg)

    st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-image: 'background_img2.jpg';
        }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        "## This is the App"


def model_summary():
    st.title("Model Summary")
    st.write("This section provides an overview of the emotion prediction model.")
    set_bg_hack("background_img3.jpg")


# Navigation bar
nav_options = ["Home", "App", "Model Summary"]
selection = st.sidebar.selectbox("**Navigation Bar**", nav_options)
#
# Display the selected section
if selection == "Home":
    home()
elif selection == "App":
    app()
elif selection == "Model Summary":
    model_summary()

#
def prediction(image_data):
  img = Image.open(image_data)
  img = img.save("img.jpg")
  img = cv2.imread("img.jpg")
  image=cv2.resize(img,(256,256))
  image=tf.constant(image,dtype=tf.float32)
  im=tf.expand_dims(image,axis=0)
  prediction=model(im)
  #prediction=tf.argmax(model(im),axis=-1).numpy()[0]
  return prediction
# st.sidebar.header('Human Emotions Detection')
# values=['Model Summary',"App","About"]
# selection=st.sidebar.selectbox("",options=values)

if selection == "App":
    # side_bg = ''
    # sidebar_bg(side_bg)
    st.subheader("Emotion Prediction App")
    st.write("Please capture or upload the image you want to analyze for facial emotions in the sidebar. ")
    capture_method = st.sidebar.radio("**:violet[Face Capture Method]**", ("Face Upload", "Camera Capture"))
    face_image = None

    if capture_method == "Face Upload":
        face_image = st.sidebar.file_uploader("**Upload Face Image**", type=['jpg', 'jpeg'])
    elif capture_method == "Camera Capture":
        face_image = st.sidebar.camera_input("Capture Face")

    CONFIGURATION = {'CLASS_NAMES': ['angry', 'happy', 'sad', 'nothing']}

    if face_image:
        st.image(face_image)
        # Replace this with your actual prediction function
        predictions = tf.argmax(prediction(face_image), axis=-1).numpy()[0]
        confidence_score = tf.reduce_max(tf.reshape(prediction(face_image), (-1,))).numpy() * 100
        confidence_score = tf.math.round(confidence_score).numpy()

        emotion = CONFIGURATION['CLASS_NAMES'][predictions]

        # Display emotion and confidence score


        # Advice and Suggestions based on predicted emotion

        if emotion == "sad":
            st.warning(f"Human Emotion: {emotion}")
            st.warning(f"Confidence score: {confidence_score}%")
            st.markdown("#### Advice")
            st.info("Remember that it's okay to feel sad sometimes. Reach out to friends or loved ones for support.")
        elif emotion == "angry":
            st.warning(f"Human Emotion: {emotion}")
            st.warning(f"Confidence score: {confidence_score}%")
            st.markdown("#### Advice")
            st.info("Take a deep breath. It's okay to feel angry, but finding healthy ways to express it can make a difference.")

        st.markdown("#### Suggestions")
        if emotion == "sad":
            st.info("Engage in activities you enjoy, listen to uplifting music, and focus on self-care.")
        elif emotion == "angry":
            st.info("Try practicing mindfulness or take a short break to calm your mind.")

        # Additional messages for the "happy" emotion
        if emotion == "happy":
            st.success(f"Human Emotion: {emotion}")
            st.success(f"Confidence score: {confidence_score}%")
            st.balloons()
            st.success("You're feeling happy! Celebrate the moment and spread positivity.")

        # Additional messages for the "nothing" emotion
        elif emotion == "nothing":
            st.error(f"Human Emotion: {emotion}")
            st.error(f"Confidence score: {confidence_score}%")
            st.error("Could not detect a human face. Please upload a clear image.")
if selection=="Model Summary":
    st.header("Emotion Recognition from Face Image")
    st.subheader("Scenario:")
    st.markdown(
        "- Imagine encountering someone in need, struggling to express their emotions."
    )
    st.markdown(
        "- They might be seething with anger, harboring resentment towards the world."
    )
    st.markdown("- Or perhaps, they're engulfed in deep sadness, weary of life.")
    st.markdown("- On the flip side, they could be brimming with happiness, yet unaware of how to savor life's joys.")
    st.write(
        "**By extending a helping hand, you become a hero, potentially preventing someone from taking drastic measures.**"
    )
    st.write("**Congratulations for being the beacon that saved a life!**")

    st.subheader("Key Questions:")
    st.markdown(
        "- How can we accurately identify emotions (angry, happy, sad) from facial images?"
    )
    st.markdown(
        "- What actionable advice, suggestions, and solutions should be provided based on detected emotions?"
    )

    st.subheader("Proposed Solution:")
    st.markdown("- Let's delve into the details of the model.")
    st.subheader("Model Objectives:")
    st.markdown(
        "- Develop a machine learning model to recognize human emotions (angry, sad, and happy) from facial images."
    )
    st.markdown(
        "- Suggest self-care activities, support networks, or mental well-being resources based on detected emotions."
    )
    st.markdown(
        "- Provide a user-friendly interface for uploading or capturing images and displaying predicted emotions with confidence scores."
    )

    st.subheader("Dataset Collection:")
    st.write(
        "Images were sourced from the internet, resulting in a diverse dataset of 6269 training files and 827 validation files."
    )
    st.write(
        "The dataset is categorized into 4 classes: angry, sad, happy, and 'nothing' (where images are not displayed properly or when the model could not detect human images)."
    )

    st.subheader("Data Preparation:")
    st.write("All images were converted to JPEG file formats.")

    st.write(
        "The tf.keras.utils.image_dataset_from_directory method was employed to generate a tf.data.Dataset from image files in a directory."
    )

    st.subheader("Data Visualization:")
    st.write("Visualizing the Training Dataset:")
    st.image("images/visualisation_train_dataset.png")
    st.write("Visualizing the Validation Dataset:")
    st.image("images/visualisation_val_dataset.png")

    st.subheader("Data Preprocessing:")
    st.write("Images were resized to a width of 256 and a height of 256.")
    st.write(
        "Resizing significantly reduces training time without compromising model performance."
    )
    st.write("Images were rescaled to a range of 0 to 1 by dividing by 255.")

    st.markdown("**Data Augmentation:**")
    st.write("Utilized tf.keras methods with the following transformations:")
    st.markdown("- RandomRotation(factor=(-0.025, 0.025))")
    st.markdown("- RandomContrast(factor=(0, 0.3))")
    st.markdown("- RandomFlip(mode='horizontal')")
    st.write("**Visualizing Data Augmentation Used:**")
    st.image("images/Augumentation.png")

    st.subheader("Modeling:")
    st.write("Both models from scratch and transfer learning models were explored.")

    st.write("**Models From Scratch:**")
    st.text("1. LeNet Model")
    st.text("2. ResNet34 Model")

    st.write("**Transfer Learning Models:**")
    st.text("1. EfficientNet B4")
    st.text("2. EfficientNet B7")
    st.text("3. ResNet50")
    st.text("4. Vgg16")
    st.text("5. Hugging Face Vit")

    st.subheader("Training:")
    st.write("Each model was trained with 40 epochs and different learning rates.")

    st.subheader("Model Validation and Testing:")
    st.write("Visualizing different models' accuracy during training.")

    st.subheader("LeNet Model")
    col1, col2 = st.columns(2)
    with col1:
        st.image("images/model_accuracy_Training_from_scratch.png")
        st.write("The model has achieved the highest accuracy of 95% on training dataset")
        st.write("The model has achieved the highest validation accuracy  of 73% on validation dataset")

    with col2:
        st.image("images/Model_loss_Training_from_scratch.png")
        st.write("The model has achieved the lowest loss of 15% on training dataset")
        st.write("The model has achieved the lowest validation loss  of 36% on validation dataset")

    st.subheader("ResNet 34 Model")
    col3,col4=st.columns(2)

    with col3:
        st.image("images/model_accuracy_Resnet34.png")
        st.write("The model has achieved the highest accuracy of 96% on training dataset")
        st.write("The model has achieved the highest validation accuracy  of 81% on validation dataset")
    with col4:
        st.image("images/model_loss_Resnet34.png")
        st.write("The model has achieved the lowest loss of 12% on training dataset")
        st.write("The model has achieved the lowest validation loss  of 60% on validation dataset")
    st.subheader("Effecient Net B4 Model")
    col5,col6=st.columns(2)

    with col5:
        st.image("images/model_accuracy_effecient_net.png")
        st.write("The model has achieved the highest accuracy of 98% on training dataset")
        st.write("The model has achieved the highest validation accuracy  of 87% on validation dataset")
    with col6:
        st.image("images/model_loss_Effecient_net.png")
        st.write("The model has achieved the lowest loss of 6% on training dataset")
        st.write("The model has achieved the lowest validation loss  of 42% on validation dataset")
    st.subheader("Effecient Net B7 Model")
    col7, col8 = st.columns(2)

    with col7:
        st.image("images/model_accuracy_effecient_net_B7.png")
        st.write("The model has achieved the highest accuracy of 98% on training dataset")
        st.write("The model has achieved the highest validation accuracy  of 88% on validation dataset")
    with col8:
        st.image("images/model_loss_Effecient_net_B7.png")
        st.write("The model has achieved the lowest loss of 6% on training dataset")
        st.write("The model has achieved the lowest validation loss  of 40% on validation dataset")

    st.subheader("Vgg16 Model")
    col11, col12 = st.columns(2)

    with col11:
        st.image("images/model_accuracy_Vgg16.png")
        st.write("The model has achieved the highest accuracy of 98% on training dataset")
        st.write("The model has achieved the highest validation accuracy  of 87% on validation dataset")
    with col12:
        st.image("images/model_loss_Vgg16.png")
        st.write("The model has achieved the lowest loss of 5% on training dataset")
        st.write("The model has achieved the lowest validation loss  of 41% on validation dataset")
    st.subheader("Resnet 50")
    col9, col10 = st.columns(2)

    with col9:
        st.image("images/model_accuracy_Resnet50.png")
        st.write("The model has achieved the highest accuracy of 98% on training dataset")
        st.write("The model has achieved the highest validation accuracy  of 87% on validation dataset")
    with col10:
        st.image("images/model_loss_Resnet50.png")
        st.write("The model has achieved the lowest loss of 4% on training dataset")
        st.write("The model has achieved the lowest validation loss  of 40% on validation dataset")

    st.subheader("Hugging Face vit google model")
    col13, col14 = st.columns(2)

    with col13:
        st.image("images/model_accuracy_hugging_face_google.png")
        st.write("The model has achieved the highest accuracy of 99% on training dataset")
        st.write("The model has achieved the highest validation accuracy  of 94% on validation dataset")
    with col14:
        st.image("images/model_loss_hugging_face_google.png")
        st.write("The model has achieved the lowest loss of 1% on training dataset")
        st.write("The model has achieved the lowest validation loss  of 25% on validation dataset")

    st.write("From the Visualisations the best performing model is Hugging Face Vit")
    st.subheader("Testing:")
    st.write("The model was tested with 16 images from the validation dataset using Hugging Face Model.")
    st.write("Results achieved:")
    st.image("images/Testing.png")
    st.write("Out of 16 images the model has made a wrong prediction in one occasion")

    st.subheader("Confusion Matrix:")
    st.write(
        "A confusion matrix is a performance evaluation tool, showcasing the accuracy of a classification model."
    )
    st.write("Model Confusion Matrix:")
    st.image("images/confusion_matrix.png")

    st.subheader("Conclusion:")
    st.write("The model achieved a remarkable 93% accuracy on the validation dataset.")
    st.write("While successful, the model remains subject to ongoing improvements and refinements.")
# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Contact Information")
st.sidebar.markdown("For inquiries, please contact uwess529300@gmail.com")
st.sidebar.markdown("© 2024 Emotion Prediction App")
