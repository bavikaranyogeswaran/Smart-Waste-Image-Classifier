"""
Prediction module for the Smart Waste Image Classifier.

This module provides the core inference logic to take an uploaded image,
preprocess it, and run it through a PyTorch model to predict its waste class.
"""
import torch
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_image(image_file, model, classes):
    """
    Predicts the class of an uploaded waste image.

    This function takes an image file, preprocesses it, and runs it
    through the provided PyTorch model to determine the most likely
    waste class and its confidence score.

    Args:
        image_file (file-like object): The uploaded image file.
        model (torch.nn.Module): The loaded PyTorch model.
        classes (list of str): List of class names mapping to model indices.

    Returns:
        dict: A dictionary containing:
            - 'class' (str): The predicted class name.
            - 'confidence' (float): The confidence score of the prediction.
    """
    # 1. Open the image file and convert it to RGB color space
    image = Image.open(image_file).convert("RGB")
    
    # 2. Apply preprocessing transforms and add batch dimension
    image_tensor = transform(image).unsqueeze(0)

    # 3. Disable gradient calculation for inference
    with torch.no_grad():
        # 4. Pass the image tensor through the model
        outputs = model(image_tensor)
        
        # 5. Calculate probabilities using softmax
        probabilities = torch.softmax(outputs, dim=1)
        
        # 6. Extract the highest probability (confidence) and its index
        confidence, predicted_idx = torch.max(probabilities, 1)

    # 7. Return the corresponding class name and confidence score
    return {
        "class": classes[predicted_idx.item()],
        "confidence": round(confidence.item(), 4)
    }
