import cv2
import time


class FPSCounter:
    def __init__(self):
        self.previous_time = time.time()

    def get_fps(self):
        current_time = time.time()
        fps = 1.0 / max(current_time - self.previous_time, 1e-6)
        self.previous_time = current_time
        return fps


def resize_keep_aspect(image, max_size=900):
    height, width = image.shape[:2]

    if max(height, width) <= max_size:
        return image

    scale = max_size / max(height, width)

    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def draw_text_background(
    image,
    text,
    position,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.6,
    text_color=(255, 255, 255),
    background_color=(45, 45, 45),
    thickness=2,
    padding=6,
):
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    x, y = position

    cv2.rectangle(
        image,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + baseline + padding),
        background_color,
        -1,
    )

    cv2.putText(
        image,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))
