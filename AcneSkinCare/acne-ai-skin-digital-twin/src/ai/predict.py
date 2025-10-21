import torch
from PIL import Image

def load_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model

def preprocess_image(image):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

def predict_skin_condition(model, image):
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

def analyze_image(image_path, model_path):
    image = Image.open(image_path).convert('RGB')
    model = load_model(model_path)
    processed_image = preprocess_image(image)
    prediction = predict_skin_condition(model, processed_image)
    return prediction