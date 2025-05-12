Number Plate Detection and Recognition

This project demonstrates how to detect and recognize number plates from images using OpenCV's Haar Cascades for object detection and EasyOCR for optical character recognition.

📌 Features Detects vehicle number plates in an image using Haar Cascade classifiers.

Recognizes text from detected number plates using EasyOCR.

Displays the image with bounding boxes and recognized text annotations.

🛠 Requirements Python 3.x

OpenCV

EasyOCR

NumPy

Google Colab (recommended for cv2_imshow support)

📦 Installation

pip install opencv-python easyocr For Colab: Make sure to run the notebook in Google Colab and use the provided cv2_imshow method for displaying images.

📁 Usage Step 1: Import and Setup python Copy Edit import cv2 import easyocr import numpy as np from google.colab.patches import cv2_imshow Step 2: Define the Detection Function

def detect_number_plate(image_path): img = cv2.imread(image_path) if img is None: print("Error loading image!") return

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
reader = easyocr.Reader(['en'])

for (x, y, w, h) in plates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plate_img = img[y:y+h, x:x+w]
    result = reader.readtext(plate_img)
    for detection in result:
        text = detection[1]
        print(f"Detected Number Plate: {text}")
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2_imshow(img)
Step 3: Run the Function

image_path = '/content/NUMPLATE.jpeg' detect_number_plate(image_path) 🖼 Example Output The script will:

Print the detected number plate text.

Display the image with the number plate region highlighted and text overlaid.

📌 Notes Detection quality depends on image clarity and angle.

You can try training a more accurate detection model (like YOLO or SSD) for better results on varied images.

📄 License This project is open-source and available under the MIT Licence
