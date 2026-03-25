import os
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMG_SIZE = 224
model = None

def load_model_once():
    global model
    if model is None:
        print("Loading model...")
        model = tf.keras.models.load_model("cnn_model.h5")
        print("Model loaded")


def preprocess_image(filepath):
    img = cv2.imread(filepath)

    if img is None:
        print("Image read failed")
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    return img


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        try:
            file = request.files.get("image")

            if not file:
                return render_template("index.html", result="No file uploaded")

            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            img = preprocess_image(filepath)

            if img is None:
                return render_template("index.html", result="Invalid image")

            load_model_once()

            pred = model.predict(img)[0][0]

            if pred < 0.5:
                label = "AI"
                confidence = (1 - pred) * 100
            else:
                label = "REAL"
                confidence = pred * 100

            if confidence < 70:
                result = f"UNCERTAIN ({confidence:.2f}%)"
            else:
                result = f"{label} ({confidence:.2f}%)"

        except Exception as e:
            print("ERROR:", str(e))
            result = f"Error: {str(e)}"

    return render_template("index.html", result=result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
