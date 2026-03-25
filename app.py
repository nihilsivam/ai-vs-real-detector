from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

IMG_SIZE = 224

# 🔥 Lazy load model (CRITICAL FIX)
model = None

def load_model_once():
    global model
    if model is None:
        print("Loading model...")
        model = tf.keras.models.load_model("cnn_model.h5")
        print("Model loaded successfully")


def preprocess_image(filepath):
    img = cv2.imread(filepath)

    if img is None:
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    return img


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files.get("image")

        if file:
            filepath = os.path.join("static", file.filename)
            file.save(filepath)

            img = preprocess_image(filepath)

            if img is not None:
                # 🔥 Load model only when needed
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
            else:
                result = "Invalid image"

    return render_template("index.html", result=result)


# 🔴 REQUIRED FOR RENDER
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
