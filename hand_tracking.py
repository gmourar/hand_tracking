import cv2
import mediapipe as mp
import numpy as np
import time

confidence = float
webcam_img = np.ndarray
rgb_tuple = tuple[int,int,int]



class Detector:
    def __init__(self,
                 mode: bool = False,
                 number_hands: int = 2,
                 model_complexity: int =  1,
                 min_detec_confidence: confidence = 0.5,
                 min_tracking_confidence: confidence = 0.5):
        self.tip_ids = [4,8,12,16,20]
        self.mp_draw = mp.solutions.drawing_utils
    
        self.mode = mode
        self.max_num_hands = number_hands
        self.complexity = model_complexity
        self.detection_conf = min_detec_confidence 
        self.tracking_conf = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode,
                                         self.max_num_hands,
                                         self.complexity,
                                         self.detection_conf,
                                         self.tracking_conf)
        

    def find_hands(self,
                   img: webcam_img,
                   draw_hands: bool = True):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # correção para rgb
        
        self.results = self.hands.process(rgb_img)

        if self.results.multi_hand_landmarks and draw_hands:
            for hand in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand, self.mp_hands.HAND_CONNECTIONS)
        return img


        
if __name__ == '__main__':
    Detec = Detector()
    capture = cv2.VideoCapture(0)
    while True:
        # captura do frame
        _ , img =  capture.read()

        # manipulação de frame
        img = Detec.find_hands(img)

        cv2.imshow('Teste Hand' , img)
        if cv2.waitKey(20) & 0xFF==ord('q'):
            break