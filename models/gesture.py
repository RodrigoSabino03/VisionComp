import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time

class Detector:
    def __init__(self,
                mode: bool=False, 
                number_hands: int=2, 
                model_complexity: int=1,
                min_detec_confidence: float=0.5,
                min_tracking_confidence: float=0.5,
                ):
        
        # parametros para iniciar o modelo do media pipe
        self.mode = mode
        self.max_num_hands = number_hands
        self.complexity = model_complexity
        self.detection_con = min_detec_confidence
        self.tracking_con = min_tracking_confidence
        self.model_path = 'gesture_recognizer.task'
        self.base_options = python.BaseOptions(model_asset_path=self.model_path)
        self.options = vision.GestureRecognizerOptions(base_options=self.base_options)
        self.recognizer = vision.GestureRecognizer.create_from_options(self.options)


        # inicializando o hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, 
                                         self.max_num_hands,
                                         self.complexity,
                                         self.detection_con,
                                         self.tracking_con,
                                         )
        
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, 
                   img: np.ndarray,
                   draw_hands: bool = True):
        #correçao de cor
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #result
        self.results = self.hands.process(img_RGB)
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                if draw_hands:
                    self.mp_draw.draw_landmarks(img, hand, self.mp_hands.HAND_CONNECTIONS)
        
        return img
    
    def find_position(
            self,
            img: np.ndarray,
            hand_number: int=0
    ):
        self.required_landmark_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[0]
            for id, lm in enumerate(my_hand.landmarks):
                height, width, _ = img.shape
                center_x = int(lm.x*width)
                center_y = int(lm.y*height)

                self.required_landmark_list.append([id, center_x, center_y])
        return self.required_landmark_list

if __name__ == '__main__':
    results = []
    images = []

    capture = cv2.VideoCapture(0)

    Detector_hands = Detector()
    

    while True: 
        _, img = capture.read()

        recognition_result = Detector_hands.recognizer.recognize(image)

        

        img = Detector_hands.find_hands(img)


        cv2.imshow("imagem do rodrigo", img)

        if cv2.waitKey(20) & 0xFF==ord('q'):
            break
