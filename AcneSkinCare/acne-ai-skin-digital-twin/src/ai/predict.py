import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# HAM10000 sınıf isimleri (Colab'daki CLASSES ile aynı sırada)
HAM10000_CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

CLASS_DESCRIPTIONS = {
    'akiec': 'Actinic Keratoses (Güneş Lekesi)',
    'bcc': 'Basal Cell Carcinoma (Bazal Hücreli Kanser)',
    'bkl': 'Benign Keratosis-like Lesions (İyi Huylu)',
    'df': 'Dermatofibroma (Deri Fibroma)',
    'mel': 'Melanoma (Kötü Huylu Kanser)',
    'nv': 'Nevus (Ben/Mole)',
    'vasc': 'Vascular Lesions (Damar Lezyonları)'
}

def load_model(model_path):
    """Keras modelini yükle"""
    # Load without compiling to avoid custom loss function issues
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

def preprocess_image(image, img_size=300):
    """EfficientNetB3 için görüntü ön işleme (300x300)"""
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')

    # Resize to model input size
    image = image.resize((img_size, img_size))

    # Convert to numpy array
    image_array = np.array(image)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # EfficientNet preprocessing
    image_array = preprocess_input(image_array)

    return image_array

def predict_skin_condition(model, processed_image):
    """HAM10000 sınıf tahmini"""
    predictions = model.predict(processed_image, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])

    predicted_class = HAM10000_CLASSES[predicted_class_idx]

    # Tüm sınıf skorları
    class_scores = {}
    for i, class_name in enumerate(HAM10000_CLASSES):
        class_scores[class_name] = float(predictions[0][i])

    return {
        'predicted_class': predicted_class,
        'predicted_class_desc': CLASS_DESCRIPTIONS[predicted_class],
        'confidence': confidence,
        'class_scores': class_scores,
        'all_predictions': predictions[0].tolist()
    }

def analyze_image(image_path, model_path):
    """Tam analiz pipeline"""
    image = Image.open(image_path).convert('RGB')
    model = load_model(model_path)
    processed_image = preprocess_image(image)
    result = predict_skin_condition(model, processed_image)
    return result

def analyze_image_from_bytes(image_bytes, model_path):
    """Bytes'dan görüntü analizi"""
    import io
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    model = load_model(model_path)
    processed_image = preprocess_image(image)
    result = predict_skin_condition(model, processed_image)
    return result