import cv2
import time
from src.tracker import HeadTracker


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Błąd: Nie można otworzyć kamery.")
        return

    tracker = HeadTracker()
    prev_time = 0

    print("System śledzenia uruchomiony. Wciśnij 'q', aby wyjść.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Odbicie lustrzane
        frame = cv2.flip(frame, 1)

        # Przetwarzanie klatki przez MediaPipe
        coords, frame = tracker.process_frame(frame)

        # Obliczanie FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # Nakładanie informacji tekstowych
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if coords:
            x, y, z = coords
            coord_text = f"X: {x:+.2f} | Y: {y:+.2f} | Z: {z:.2f} m"
            cv2.putText(frame, coord_text, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "BRAK DETEKCJI", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Wirtualne Okno - Head Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()