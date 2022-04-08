try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

# If you don't have tesseract executable in your PATH, include the following:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
# Example tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'


text = pytesseract.image_to_string(Image.open(r'images\0.jpg'))
print(f'Text: {text}')

# Simple image to string
# print(pytesseract.image_to_string(Image.open(r'images\60.jpg')))