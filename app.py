import cv2
import pytesseract

img = cv2.imread("image.jpg")

text = pytesseract.image_to_string(img)

print(text)
