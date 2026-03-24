from flask import Flask, render_template, request
import cv2
import numpy as np
import joblib
from skimage.feature import local_binary_pattern
import os

app = Flask(__name__)

# Load model + scaler
model = joblib.load("ai_detector_model.pkl")
scaler = joblib.load("scaler.pkl")

# Feature settings
radius = 1
n_points = 8 * radius

def extract_features(img_path):
    img = cv2.imread(img_path)

    if img is None:
        return None

    img = cv2.resize(img, (256, 256))

    hist = cv2.calcHist([img], [0,1,2], None, [8,8,8], [0,256]*3)
    hist = cv2.normalize(hist, hist).flatten()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points+3),
        range=(0, n_points+2)
    )
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / (256*256)

    # FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)

    fft_mean = np.mean(magnitude)
    fft_std = np.std(magnitude)

    features = np.hstack([hist, lbp_hist, edge_density, fft_mean, fft_std])

    return features


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            filepath = os.path.join("static", file.filename)
            file.save(filepath)

            features = extract_features(filepath)

            if features is not None:
                features = features.reshape(1, -1)
                features = scaler.transform(features)

                pred = model.predict(features)[0]
                prob = model.predict_proba(features)[0]

                label = "AI" if pred == 0 else "REAL"
                confidence = max(prob) * 100

                if confidence < 70:
                    result = f"UNCERTAIN ({confidence:.2f}%)"
                else:
                    result = f"{label} ({confidence:.2f}%)"

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
