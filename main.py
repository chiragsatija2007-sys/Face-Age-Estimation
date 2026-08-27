import os
import cv2

from src.face_detector import FaceDetector
from src.age_detector import AgeDetector
from src.dataset_loader import DatasetLoader
from src.ui import draw_header, draw_footer, draw_face
from src.utils import FPSCounter


def main():
    print("=" * 50)
    print(" AI FACE AGE DETECTION SYSTEM ")
    print("=" * 50)

    metadata_path = os.path.join("data", "metadata.csv")

    if os.path.exists(metadata_path):
        loader = DatasetLoader(metadata_path)
        loader.load()
        print(f"[INFO] Dataset Records : {loader.total_images()}")

    print("[INFO] Loading Face Detector...")
    face_detector = FaceDetector()

    print("[INFO] Loading Age Detector...")
    age_detector = AgeDetector()

    print("[INFO] Models Loaded Successfully.")

    fps_counter = FPSCounter()
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("[ERROR] Unable to open webcam.")
        return

    print("\nPress Q to Quit.\n")

    frame_count = 0
    PROCESS_EVERY_N_FRAMES = 3  
    cached_faces = []
    cached_labels = {}

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame_count += 1
        fps = fps_counter.get_fps()

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            cached_faces = face_detector.detect_faces(
                frame,
                confidence_threshold=0.6
            )
            
            if len(cached_faces) == 0:
                age_detector.reset()
                cached_labels.clear()
            else:
                for index, face in enumerate(cached_faces, start=1):
                    x, y, w, h = face["box"]
                    padding = max(5, int(min(w, h) * 0.03))

                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    y2 = min(frame.shape[0], y + h + padding)

                    face_roi = frame[y1:y2, x1:x2]

                    if face_roi.size > 0:
                        result = age_detector.predict(face_roi)
                        cached_labels[index] = f"{result['label']} ({result['confidence']:.1f}%)"

        draw_header(frame, "AI Face Age Detection System", len(cached_faces))

        for index, face in enumerate(cached_faces, start=1):
            x, y, w, h = face["box"]
            age_label = cached_labels.get(index, "Processing...")
            
            draw_face(frame, x, y, w, h, index, age_label)

        cv2.putText(
            frame,
            f"FPS : {fps:.1f}",
            (frame.shape[1] - 120, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        draw_footer(frame)
        cv2.imshow("AI Face Age Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    print("[INFO] Application Closed.")


if __name__ == "__main__":
    main()
