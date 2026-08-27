import os
import cv2
import tkinter as tk
from tkinter import filedialog

from src.face_detector import FaceDetector
from src.age_detector import AgeDetector
from src.ui import draw_header, draw_footer, draw_face
from src.utils import resize_keep_aspect


def main():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    print("=" * 50)
    print(" AI FACE AGE DETECTION SYSTEM ")
    print("=" * 50)
    print("[INFO] Loading models...")

    face_detector = FaceDetector()
    age_detector = AgeDetector()
    print("[INFO] Models Loaded Successfully.")

    project_root = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(project_root, "test")
    os.makedirs(test_dir, exist_ok=True)

    while True:
        file_path = filedialog.askopenfilename(
            title="Select Test Image",
            initialdir=test_dir,
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )

        if not file_path:
            print("[INFO] No image selected.")
            break

        image = cv2.imread(file_path)

        if image is None:
            print("[ERROR] Unable to open image.")
            continue

        image = resize_keep_aspect(image, 900)

        age_detector.reset()

        faces = face_detector.detect_faces(image, confidence_threshold=0.6)
        draw_header(image, os.path.basename(file_path), len(faces))

        for index, face in enumerate(faces, start=1):
            x, y, w, h = face["box"]
            padding = max(5, int(min(w, h) * 0.03))

            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)

            roi = image[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            result = age_detector.predict(roi)

            age_text = f"{result['label']} ({result['confidence']:.1f}%)"

            draw_face(image, x, y, w, h, index, age_text)
            print(f"Face {index}: {age_text}")

        draw_footer(image)
        cv2.imshow("AI Face Age Detection", image)
        print("[INFO] Press any key for another image or Q to Quit.")

        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        if key == ord("q") or key == ord("Q"):
            break

    print("[INFO] Application Closed.")


if __name__ == "__main__":
    main()
