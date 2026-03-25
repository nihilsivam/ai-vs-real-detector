from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load model once
model = tf.keras.models.load_model("cnn_model.h5")

IMG_SIZE = 224


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
        file = request.files["image"]

        if file:
            filepath = os.path.join("static", file.filename)
            file.save(filepath)

            img = preprocess_image(filepath)

            if img is not None:
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

    return render_template("index.html", result=result)


# 🔴 REQUIRED FOR RENDER DEPLOYMENT
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
