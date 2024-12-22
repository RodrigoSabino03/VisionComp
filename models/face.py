import cv2
import mediapipe as mp
import numpy as np

class FaceMeshDetector:
    def __init__(self,
                 max_num_faces: int = 1, 
                 min_detection_confidence: float = 0.5, 
                 min_tracking_confidence: float = 0.5):
        
        # Inicializa os parâmetros e o FaceMesh do Mediapipe
        self.max_num_faces = max_num_faces
        self.detection_confidence = min_detection_confidence
        self.tracking_confidence = min_tracking_confidence
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=self.max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.my_drawing_specs = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)

    def find_faces(self, img: np.ndarray, draw_faces: bool = True):
        # Processa a imagem para encontrar landmarks faciais
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        
        if results.multi_face_landmarks:
            for face_landmark in results.multi_face_landmarks:
                if draw_faces:
                    # Desenha os contornos e a malha do rosto
                    self.mp_drawing.draw_landmarks(
                        img,
                        face_landmark,
                        self.mp_face_mesh.FACEMESH_TESSELATION, 
                        None, 
                        self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    self.mp_drawing.draw_landmarks(
                        img,
                        face_landmark,
                        self.mp_face_mesh.FACEMESH_CONTOURS, 
                        None,
                        self.my_drawing_specs
                    )
        return img

    def get_landmarks(self, img: np.ndarray, face_number: int = 0):
        landmarks = []
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)

        if results.multi_face_landmarks and len(results.multi_face_landmarks) > face_number:
            face_landmark = results.multi_face_landmarks[face_number]
            for id, lm in enumerate(face_landmark.landmarks):
                height, width, _ = img.shape
                center_x = int(lm.x * width)
                center_y = int(lm.y * height)
                landmarks.append([id, center_x, center_y])
        
        return landmarks


if __name__ == '__main__':
    capture = cv2.VideoCapture(0)
    face_detector = FaceMeshDetector()

    while True:
        _, img = capture.read()

        # Encontra as faces e desenha os landmarks
        img = face_detector.find_faces(img)

        cv2.imshow("imagem do rodrigo", img)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()
