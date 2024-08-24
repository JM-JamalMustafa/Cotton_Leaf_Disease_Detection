# Cotton_Leaf_Disease_Detection
This project is a web-based application for detecting diseases in cotton leaves and plants using a deep learning model. The model classifies images into one of four categories: diseased cotton leaf, diseased cotton plant, fresh cotton leaf, or fresh cotton plant.

# Features
Deep Learning Model: Utilizes a pre-trained InceptionV3 model, fine-tuned for cotton leaf disease detection.
Flask Web Framework: The application is built using Flask, a lightweight web framework for Python.
Image Preprocessing: Images are preprocessed using TensorFlow's Keras utilities, including resizing and normalization.
GPU Support: The application is configured to utilize GPU resources efficiently, allowing for faster predictions.
File Upload Handling: Users can upload images through the web interface, and the application will display the prediction results.
Files
app.py: The main file containing the Flask application.
model_inceptionv3.h5: The pre-trained model used for prediction.
index.html: The main page template for the web application.
# Installation
Clone the repository:
git clone https://github.com/yourusername/cotton-leaf-disease-detection.git
cd cotton-leaf-disease-detection
# Install the required dependencies:
pip install -r requirements.txt

Download the model file: Place the model_inceptionv3.h5 file in the project directory as specified in MODEL_PATH.

# Run the application:

python app.py
Access the web application: Open your browser and go to http://127.0.0.1:5001/.

Usage
![results1](https://github.com/user-attachments/assets/a66f1290-0dff-435f-bbdb-c6eb6178b118)
Upload an image of a cotton leaf or plant through the web interface.
The model will process the image and predict whether it is a diseased cotton leaf, diseased cotton plant, fresh cotton leaf, or fresh cotton plant.
The result will be displayed on the same page.
Model Details
The model was trained using the InceptionV3 architecture, which is known for its high accuracy in image classification tasks. It was fine-tuned on a dataset of cotton leaf and plant images to specifically detect diseases.

GPU Configuration
The application is optimized to run on a GPU, making use of TensorFlow's configuration options to allocate a fraction of GPU memory and allow for dynamic growth of GPU usage.

Contributing
Feel free to contribute to this project by submitting issues or pull requests.
