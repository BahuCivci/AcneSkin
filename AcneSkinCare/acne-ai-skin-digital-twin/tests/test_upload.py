import pytest
from src.pages.upload import upload_image

def test_upload_image_success(mocker):
    mock_file = mocker.Mock()
    mock_file.name = "test_image.jpg"
    result = upload_image(mock_file)
    assert result is not None
    assert result['filename'] == "test_image.jpg"
    assert result['status'] == "success"

def test_upload_image_failure(mocker):
    mock_file = None
    result = upload_image(mock_file)
    assert result is None or result['status'] == "error"