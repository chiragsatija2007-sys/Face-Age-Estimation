import os
import urllib.request
from collections import deque

import cv2
import numpy as np


class AgeDetector:

    MODEL_NAME = "age_googlenet.onnx"

    MODEL_URL = (
        "https://github.com/onnx/models/raw/main/"
        "validated/vision/body_analysis/age_gender/"
        "models/age_googlenet.onnx"
    )

    AGE_BUCKETS = [
        "(0-2) Baby",
        "(4-6) Child",
        "(8-12) Pre-Teen",
        "(15-20) Teenager",
        "(25-32) Young Adult",
        "(38-43) Adult",
        "(48-53) Middle Aged",
        "(60-100) Senior"
    ]

    MODEL_MEAN_VALUES = (
        78.4263377603,
        87.7689143744,
        114.895847746
    )

    AGE_CALIBRATION_BOOST = {
        4: 1.25,  
        3: 0.85,  
    }

    MIN_CONFIDENCE_THRESHOLD = 35.0  

    def __init__(self):
        model_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
            "models"
        )

        os.makedirs(model_dir, exist_ok=True)

        self.model_path = os.path.join(
            model_dir,
            self.MODEL_NAME
        )

        self._download_model()

        self.net = cv2.dnn.readNetFromONNX(
            self.model_path
        )

        self.history = deque(maxlen=8)

    def _download_model(self):
        if os.path.exists(self.model_path):
            return

        print("[INFO] Downloading Age Model...")
        urllib.request.urlretrieve(self.MODEL_URL, self.model_path)
        print("[INFO] Age model downloaded.")

    def reset(self):
        self.history.clear()

    def predict(self, face):
        if (
            face is None
            or face.size == 0
            or face.shape[0] < 15
            or face.shape[1] < 15
        ):
            return {
                "label": "Unknown",
                "confidence": 0.0,
                "index": -1
            }

        blob = cv2.dnn.blobFromImage(
            face,
            1.0,
            (227, 227),
            self.MODEL_MEAN_VALUES,
            swapRB=True
        )

        self.net.setInput(blob)
        output = self.net.forward()

        probs = output[0]

        calibrated_probs = probs.copy()
        for idx, boost in self.AGE_CALIBRATION_BOOST.items():
            calibrated_probs[idx] *= boost

        calibrated_probs = calibrated_probs / calibrated_probs.sum()

        index = int(np.argmax(calibrated_probs))
        confidence = float(calibrated_probs[index] * 100)

        if confidence < self.MIN_CONFIDENCE_THRESHOLD:
            return {
                "label": "Unable to Determine",
                "confidence": confidence,
                "index": -1
            }

        self.history.append(index)
        
        stable_index = max(
            set(self.history),
            key=self.history.count
        )

        stable_confidence = float(
            calibrated_probs[stable_index] * 100
        )

        return {
            "label": self.AGE_BUCKETS[stable_index],
            "confidence": stable_confidence,
            "index": stable_index
        }

    def predict_text(self, face):
        result = self.predict(face)
        return f"{result['label']} ({result['confidence']:.1f}%)"
