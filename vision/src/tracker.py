import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HeadTracker:
    def __init__(self, model_path="face_landmarker.task"):
        if not os.path.exists(model_path):
            print("Pobieranie modelu FaceLandmarker...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("Model pobrany pomyślnie.")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.timestamp_ms = 0

        # Indeksy punktów w MediaPipe FaceLandmarker:
        # 168: punkt między brwiami (nasada nosa - nasz punkt (0,0,0) między oczami!)
        # 1:   czubek nosa
        # 33:  zewnętrzny kącik lewego oka
        # 263: zewnętrzny kącik prawego oka
        # 61:  lewy kącik ust
        # 291: prawy kącik ust
        self.PNP_INDICES = [168, 1, 33, 263, 61, 291]

        # Indeksy tęczówek (Opcja 3 - punkty wizualne oczu)
        self.LEFT_IRIS_IDX = 468
        self.RIGHT_IRIS_IDX = 473

        self.focal_scale = 0.65  # Mnożnik ogniskowej kamery (kalibracja pod FOV kamerki)

        # Uproszczony model 3D twarzy w metrach (środek układu = punkt 168 między oczami)
        # Osie: X w prawo, Y w górę, Z w stronę kamery (do przodu twarzy)
        self.model_points_3d = np.array([
            (0.0, 0.0, 0.0),  # 168: Środek między oczami
            (0.0, -0.035, 0.028),  # 1:   Czubek nosa
            (-0.042, -0.005, -0.015),  # 33:  Zewnętrzny kącik lewego oka (8.4 cm rozstawu)
            (0.042, -0.005, -0.015),  # 263: Zewnętrzny kącik prawego oka
            (-0.024, -0.072, 0.005),  # 61:  Lewy kącik ust
            (0.024, -0.072, 0.005),  # 291: Prawy kącik ust
        ], dtype=np.float64)

        # Wektor zniekształceń obiektywu (zakładamy brak dystorsji dla uproszczenia)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    def process_frame(self, frame):
        """
        Zwraca:
        - coords: (x, y, z) w metrach względem środka kamery (lub None)
        - frame: klatka z wizualizacją punktów i wektora spojrzenia
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        self.timestamp_ms += 33
        detection_result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)

        h, w, _ = frame.shape
        coords = None

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]

            # 1. Przygotowanie macierzy wewnętrznej kamery z uwzględnieniem kalibracji
            focal_length = float(w) * self.focal_scale
            center = (w / 2.0, h / 2.0)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)

            # 2. Pobranie punktów 2D z klatki dla solvePnP
            image_points_2d = np.array([
                (landmarks[idx].x * w, landmarks[idx].y * h)
                for idx in self.PNP_INDICES
            ], dtype=np.float64)

            # 3. Rozwiązanie problemu Perspective-n-Point (PnP)
            success, rvec, tvec = cv2.solvePnP(
                self.model_points_3d,
                image_points_2d,
                camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                # tvec to pozycja środka między oczami (punkt 168) w metrach
                # Dopasowujemy znaki osi do układu Unity (X prawo, Y góra, Z odległość)
                pos_x = float(tvec[0][0])
                pos_y = float(-tvec[1][0])  # Odwracamy Y, bo w OpenCV oś Y rośnie w dół
                pos_z = float(tvec[2][0])

                coords = (pos_x, pos_y, pos_z)

                # --- WIZUALIZACJA (Opcja 3: tęczówki + punkt między oczami) ---
                # Rysowanie 6 punktów modelu PnP (zielone kropki)
                for pt in image_points_2d:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

                # Tęczówki (niebieskie kółka) i wyliczony środek oczu (czerwony punkt)
                if len(landmarks) > self.RIGHT_IRIS_IDX:
                    l_iris = landmarks[self.LEFT_IRIS_IDX]
                    r_iris = landmarks[self.RIGHT_IRIS_IDX]

                    l_px = (int(l_iris.x * w), int(l_iris.y * h))
                    r_px = (int(r_iris.x * w), int(r_iris.y * h))
                    eye_center_px = ((l_px[0] + r_px[0]) // 2, (l_px[1] + r_px[1]) // 2)

                    cv2.circle(frame, l_px, 4, (255, 100, 0), -1)
                    cv2.circle(frame, r_px, 4, (255, 100, 0), -1)
                    cv2.circle(frame, eye_center_px, 5, (0, 0, 255), -1)

                # Rzutowanie osi 3D z punktu między oczami (pokazuje orientację głowy!)
                axis_3d = np.float32([[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]])
                img_pts, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, self.dist_coeffs)
                origin = (int(image_points_2d[0][0]), int(image_points_2d[0][1]))

                cv2.line(frame, origin, (int(img_pts[0][0][0]), int(img_pts[0][0][1])), (0, 0, 255), 2)  # Oś X
                cv2.line(frame, origin, (int(img_pts[1][0][0]), int(img_pts[1][0][1])), (0, 255, 0), 2)  # Oś Y
                cv2.line(frame, origin, (int(img_pts[2][0][0]), int(img_pts[2][0][1])), (255, 0, 0), 2)  # Oś Z

        return coords, frame

    def close(self):
        self.landmarker.close()