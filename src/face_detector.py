import os
import urllib.request
import cv2
import numpy as np


class FaceDetector:

    MODEL_URL = (
        "https://raw.githubusercontent.com/Linzaer/"
        "Ultra-Light-Fast-Generic-Face-Detector-1MB/"
        "master/models/onnx/version-RFB-320.onnx"
    )

    MODEL_NAME = "version-RFB-320.onnx"

    def __init__(self):
        self.model_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
            "models"
        )

        os.makedirs(self.model_dir, exist_ok=True)

        self.model_path = os.path.join(
            self.model_dir,
            self.MODEL_NAME
        )

        self._download_model()

        self.net = cv2.dnn.readNetFromONNX(
            self.model_path
        )

    def _download_model(self):
        if os.path.exists(self.model_path):
            return

        print("[INFO] Downloading Face Detection Model...")

        urllib.request.urlretrieve(
            self.MODEL_URL,
            self.model_path
        )

        print("[INFO] Face model downloaded.")

    def detect_faces(
        self,
        image,
        confidence_threshold=0.6
    ):
        h, w = image.shape[:2]

        resized = cv2.resize(image, (320, 240))

        rgb = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB
        )

        rgb = (rgb - 127.0) / 128.0

        blob = np.transpose(
            rgb,
            (2, 0, 1)
        )

        blob = np.expand_dims(
            blob,
            axis=0
        ).astype(np.float32)

        self.net.setInput(blob)

        scores, boxes = self.net.forward(
            ["scores", "boxes"]
        )

        scores = scores[0]
        boxes = boxes[0]

        faces = []

        for i in range(scores.shape[0]):
            confidence = float(scores[i][1])

            if confidence < confidence_threshold:
                continue

            box = boxes[i]

            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(w, x2)
            y2 = min(h, y2)

            face_w = x2 - x1
            face_h = y2 - y1

            if face_w < 20 or face_h < 20:
                continue

            faces.append({
                "box": (
                    x1,
                    y1,
                    face_w,
                    face_h
                ),
                "confidence": confidence
            })

        faces.sort(
            key=lambda f: f["confidence"],
            reverse=True
        )

        return faces
