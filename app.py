import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
from torch import nn

# Device Configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load Trained Model
@st.cache_resource
def load_model():
    # Define the model architecture
    model = models.resnet18(pretrained=True)
    
    # Custom fully connected layer
    class Dc_model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(512, 120)
            self.linear2 = nn.Linear(120, 2)

        def forward(self, x):
            x = F.relu(self.linear1(x))
            x = self.linear2(x)
            return x

    model.fc = Dc_model()  # Replace the last layer
    model.load_state_dict(torch.load("Model\cat_dog_best.pth", map_location=device))  # Load your trained weights
    model.eval()  # Set to evaluation mode
    model.to(device)
    return model

# Image Preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((280, 280)),  # Resize to match model input size
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

# Prediction Function
def predict_image(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = probabilities.argmax(dim=1).item()
        confidence = probabilities.max().item()
    return predicted_class, confidence

# Streamlit App
def main():
    st.title("Cat vs Dog Classifier")
    st.write("Upload an image of a cat or a dog, and the model will predict the class.")

    # Upload Image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess Image
        st.write("Processing image...")
        image_tensor = preprocess_image(image)

        # Load the model
        model = load_model()

        # Make Prediction
        st.write("Classifying...")
        class_names = ["Cat", "Dog"]
        predicted_class, confidence = predict_image(model, image_tensor)

        # Display Results
        st.write(f"Prediction: **{class_names[predicted_class]}**")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")

if __name__ == "__main__":
    main()
