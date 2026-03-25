from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

model = tf.keras.models.load_model("cnn_model.h5")
IMG_SIZE = 224

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            filepath = os.path.join("static", file.filename)
            file.save(filepath)

            img = cv2.imread(filepath)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

            pred = model.predict(img)[0][0]

            if pred < 0.5:
                result = "AI"
            else:
                result = "REAL"

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)interface.launch()
