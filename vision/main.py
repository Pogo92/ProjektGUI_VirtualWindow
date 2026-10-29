import cv2
import time

def main():
    # 0 to domyślna wbudowana/pierwsza kamera; zmień na 1 lub 2, jeśli używasz zewnętrznej
    cap = cv2.VideoCapture(0)

    # Opcjonalnie wymuszenie rozdzielczości i klatkażu kamery
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Błąd: Nie można otworzyć kamery.")
        return

    prev_time = 0

    print("Kamera uruchomiona pomyślnie. Wciśnij 'q', aby zakończyć.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Błąd: Nie udało się pobrać klatki.")
            break

        # Odbicie lustrzane obrazu (niezbędne, by ruch głowy w prawo był ruchem w prawo na ekranie)
        frame = cv2.flip(frame, 1)

        # Wyliczenie aktualnego FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # Wyświetlenie FPS na obrazie
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Wirtualne Okno - Test Kamery", frame)

        # Wyjście po wciśnięciu klawisza 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()