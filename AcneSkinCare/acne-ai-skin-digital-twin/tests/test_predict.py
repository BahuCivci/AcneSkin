import pytest
from src.ai.predict import predict_skin_condition

def test_predict_skin_condition_valid_image():
    # Assuming we have a valid image path for testing
    image_path = "tests/test_images/valid_image.jpg"
    result = predict_skin_condition(image_path)
    assert result is not None
    assert isinstance(result, dict)
    assert "condition" in result
    assert "score" in result

def test_predict_skin_condition_invalid_image():
    image_path = "tests/test_images/invalid_image.txt"
    result = predict_skin_condition(image_path)
    assert result is None

def test_predict_skin_condition_empty_image():
    image_path = "tests/test_images/empty_image.jpg"
    result = predict_skin_condition(image_path)
    assert result is None