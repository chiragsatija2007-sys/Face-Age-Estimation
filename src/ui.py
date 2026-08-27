import cv2

THEME_PRIMARY = (147, 20, 255)
THEME_DARK = (10, 10, 10)
THEME_TEXT = (255, 255, 255)
THEME_ACCENT = (200, 100, 255)

def draw_face(frame, x, y, w, h, index, age_text):
    line_len = max(15, int(w * 0.15))
    thickness = 2

    cv2.line(frame, (x, y), (x + line_len, y), THEME_PRIMARY, thickness)
    cv2.line(frame, (x, y), (x, y + line_len), THEME_PRIMARY, thickness)
    
    cv2.line(frame, (x + w, y), (x + w - line_len, y), THEME_PRIMARY, thickness)
    cv2.line(frame, (x + w, y), (x + w, y + line_len), THEME_PRIMARY, thickness)
    
    cv2.line(frame, (x, y + h), (x + line_len, y + h), THEME_PRIMARY, thickness)
    cv2.line(frame, (x, y + h), (x, y + h - line_len), THEME_PRIMARY, thickness)
    
    cv2.line(frame, (x + w, y + h), (x + w - line_len, y + h), THEME_PRIMARY, thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - line_len), THEME_PRIMARY, thickness)

    label = f"ID:{index} | {age_text}"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.5
    font_thickness = 1

    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    
    bg_y1 = max(0, y - text_h - 10)
    bg_y2 = bg_y1 + text_h + 10
    bg_x2 = x + text_w + 10

    cv2.rectangle(frame, (x, bg_y1), (bg_x2, bg_y2), THEME_DARK, cv2.FILLED)
    cv2.line(frame, (x, bg_y1), (bg_x2, bg_y1), THEME_PRIMARY, 2)
    cv2.putText(frame, label, (x + 5, bg_y2 - 5), font, font_scale, THEME_TEXT, font_thickness, cv2.LINE_AA)


def draw_header(frame, title, face_count):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), THEME_DARK, cv2.FILLED)
    cv2.line(frame, (0, 40), (frame.shape[1], 40), THEME_PRIMARY, 2)

    cv2.putText(frame, f"> {title.upper()}", (15, 27), 
                cv2.FONT_HERSHEY_DUPLEX, 0.6, THEME_PRIMARY, 1, cv2.LINE_AA)
    
    status_text = f"TARGETS DETECTED: {face_count}"
    status_color = THEME_ACCENT if face_count > 0 else (100, 100, 100)
    
    (text_w, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
    cv2.putText(frame, status_text, (frame.shape[1] - text_w - 15, 27), 
                cv2.FONT_HERSHEY_DUPLEX, 0.5, status_color, 1, cv2.LINE_AA)


def draw_footer(frame):
    footer_y = frame.shape[0] - 30
    
    cv2.rectangle(frame, (0, footer_y), (frame.shape[1], frame.shape[0]), THEME_DARK, cv2.FILLED)
    cv2.putText(frame, "Press 'Q' to Exit | AI Processing Active", (15, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
