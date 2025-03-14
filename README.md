# handwritten_digit_recognition
# Overview -->
           This project is a web-based digit recognition system where users upload images of handwritten digits (0-9), and a trained neural network predicts the digit.
# Workflow -->
  1. Model Training -
             The MNIST dataset is loaded, normalized, and trained on a neural network model with three dense layers.
             The model is compiled using the Adam optimizer and trained for three epochs.
  2. Flask API -
             A Flask server is created with an upload feature.
             Users upload digit images through a simple frontend.
             The server processes the image (resizes, inverts colors, normalizes).
             The model predicts the digit and returns the result as JSON.
  3. Frontend -
            A basic webpage allows users to upload an image.
            JavaScript sends the image to the Flask backend via an API request.
            The predicted digit is displayed on the webpage.
  4. Execution -
            Run the Flask app and access it via a browser.
            Upload a digit image and receive the prediction.
