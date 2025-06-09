# requirements:
#   pip install opencv-python pytesseract sqlite3 (builtin) python-dateutil
#   Install Tesseract OCR separately (add it to your PATH).

import cv2
import pytesseract
import sqlite3
from datetime import datetime
from dateutil import tz          # nice timezone handling

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
# If Tesseract isn’t on PATH, uncomment and set the correct path:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
DB_FILE = "parking.db"
TIMEZONE = tz.gettz("America/New_York")   # pick yours

# ---------------------------------------------------------------------
# SQLite helper
# ---------------------------------------------------------------------
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS parking_sessions (
            id         INTEGER PRIMARY KEY,
            plate      TEXT    NOT NULL,
            entry_time TEXT    NOT NULL,
            exit_time  TEXT    NULL
        )
        """)
        conn.commit()

def log_event(plate: str):
    """Decide entry vs exit and write to DB."""
    now = datetime.now(TIMEZONE).isoformat(timespec="seconds")

    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()

        # Is there an *open* session (exit_time is NULL) for this plate?
        c.execute("""
            SELECT id FROM parking_sessions
            WHERE plate = ? AND exit_time IS NULL
            ORDER BY entry_time DESC LIMIT 1
        """, (plate,))
        row = c.fetchone()

        if row:                       # ----> EXIT
            c.execute(
                "UPDATE parking_sessions SET exit_time = ? WHERE id = ?",
                (now, row[0])
            )
            print(f"[{now}] EXIT  - {plate}")
        else:                         # ----> ENTRY
            c.execute(
                "INSERT INTO parking_sessions (plate, entry_time) VALUES (?, ?)",
                (plate, now)
            )
            print(f"[{now}] ENTRY - {plate}")

        conn.commit()

# ---------------------------------------------------------------------
# Simple plate recognizer (same idea you used earlier)
# ---------------------------------------------------------------------
def recognize_plate(image):
    """Return the best-guess plate text, or None if not found."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blur, 30, 200)

    contours, _ = cv2.findContours(
        edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    plate_roi = None
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)
        if len(approx) == 4:          # likely rectangle
            x, y, w, h = cv2.boundingRect(approx)
            plate_roi = gray[y:y+h, x:x+w]
            break

    if plate_roi is None:
        return None

    text = pytesseract.image_to_string(
        plate_roi, config="--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )
    plate = "".join(ch for ch in text if ch.isalnum())
    return plate if 5 <= len(plate) <= 10 else None  # crude filter

# ---------------------------------------------------------------------
# Main loop (use video or still images)
# ---------------------------------------------------------------------
def run_camera(index=0):
    cap = cv2.VideoCapture(index)
    assert cap.isOpened(), "Cannot open camera"

    seen_recently = set()      # debounce same plate within X seconds
    DEBOUNCE_SEC = 15
    last_seen = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            plate = recognize_plate(frame)
            if plate:
                now_sec = datetime.now().timestamp()
                if (plate not in seen_recently or
                        now_sec - last_seen.get(plate, 0) > DEBOUNCE_SEC):
                    log_event(plate)
                    seen_recently.add(plate)
                    last_seen[plate] = now_sec

            # Show camera feed (optional)
            cv2.imshow("Parking Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

# ---------------------------------------------------------------------
# Utilities for querying the DB
# ---------------------------------------------------------------------
def list_open_sessions():
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT plate, entry_time
            FROM parking_sessions
            WHERE exit_time IS NULL
            ORDER BY entry_time
        """)
        return c.fetchall()

def list_all_sessions(limit=20):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT plate, entry_time, exit_time
            FROM parking_sessions
            ORDER BY entry_time DESC
            LIMIT ?
        """, (limit,))
        return c.fetchall()

# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    init_db()
    # run_camera()                # live camera version
    # --- or, for quick testing on a single image ---
    img = cv2.imread("car.jpg")
    plate = recognize_plate(img)
    if plate:
        log_event(plate)
    else:
        print("No plate found")
