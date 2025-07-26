import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

# Create a simple CNN model (in practice, you'd want to train this on iris flower images)
def create_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(3, activation='softmax')  # 3 classes for Iris species
    ])
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# Load or train model (in a real scenario, you'd train this first)
model = create_model()

# For demonstration, we'll use random weights - in practice you need to train the model
# model.load_weights('iris_model_weights.h5')  # Uncomment if you have trained weights

def predict_iris_species(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    
    # Map to species
    species = ['Iris setosa', 'Iris versicolor', 'Iris virginica'][predicted_class]
    flower_type = "Perennial flowering plant"  # All irises are perennial
    
    return species, flower_type

# Example usage
if __name__ == "__main__":
    img_path = input("Enter path to your iris flower image: ")
    if os.path.exists(img_path):
        species, flower_type = predict_iris_species(img_path)
        print(f"\nPredicted Species: {species}")
        print(f"Flower Type: {flower_type}")
    else:
        print("Image file not found!")