import io
import cv2
from zipfile import ZipFile

def encode_images(images):
    stream = io.BytesIO()
    with ZipFile(stream, 'w') as zf:
        for title,img in images:
            zf.writestr(title, get_response_image(img))
    stream.seek(0)
    return stream

def get_response_image(img):
    _, encoded_img = cv2.imencode('.png', img)
    return encoded_img.tobytes()