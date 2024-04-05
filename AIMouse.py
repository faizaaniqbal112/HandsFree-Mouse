import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

cam_width, cam_height = 640, 480
screen_width, screen_height = pyautogui.size()
detectionArea = 120


smoothening = 10
previousX, previousY = 0, 0  
currentX, currentY = 0, 0

#sets the webcam
cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

gesture = "None"
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (0, 0, 255) 
thickness = 2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


pTime = 0
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1) #flips the frame
    
    if not ret:
        break
    
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(image_rgb)
    
    fingers = []
    fingerLandmarks = []
    #Draw the landmakrs on the hand showing the detection
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            

            #X and Y of Index Tip and Pip
            index_finger_x6 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x
            index_finger_y6 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y 
            index_finger_x8 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
            index_finger_y8 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y 
            
            #X and Y of Middle Finger Tip and Pip
            middle_finger_x10 = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x
            middle_finger_y10 = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y 
            middle_finger_x12 = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
            middle_finger_y12 = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y 
            
            #X and Y of Ring Finger Tip and Pip
            ring_finger_x14 = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x
            ring_finger_y14 = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y 
            ring_finger_x16 = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x
            ring_finger_y16 = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y 

            #X and Y of Pinky Finger Tip and Pip
            pinky_finger_x18 = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x
            pinky_finger_y18 = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y 
            pinky_finger_x20 = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x
            pinky_finger_y20 = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y 

            #X and Y of thumb Tip and IP
            thumb_x3 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x
            thumb_y3 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y
            thumb_x4 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
            thumb_y4 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
            
            #Array containing all the necessary landmarks
            fingerLandmarks = [[index_finger_x6, index_finger_y6, index_finger_x8, index_finger_y8],
                               [middle_finger_x10, middle_finger_y10, middle_finger_x12, middle_finger_y12],
                               [ring_finger_x14, ring_finger_y14, ring_finger_x16, ring_finger_y16],
                               [pinky_finger_x18, pinky_finger_y18, pinky_finger_x20, pinky_finger_y20],
                               [thumb_x3, thumb_y3, thumb_x4, thumb_y4]]
            
            #Get the hand type
            for hand in results.multi_handedness:
                handType = hand.classification[0].label
                
                #print(handType)
                #thumb detection to see if opened or closed
                if handType == "Left": 
                    if fingerLandmarks[4][0] < fingerLandmarks[4][2]:
                        fingers.append(1)
                    else:
                         fingers.append(0)
                elif handType == "Right":
                    if fingerLandmarks[4][0] > fingerLandmarks[4][2]:
                        fingers.append(1)
                    else:
                         fingers.append(0)
            
            #finger detection to see if opened or closed
            for id in range(0,4):
                if fingerLandmarks[id][1] > fingerLandmarks[id][3]:
                    fingers.append(1)
                elif fingerLandmarks[id][1] < fingerLandmarks[id][3]:
                    fingers.append(0) 
            
            x1 = fingerLandmarks[0][2] * cam_width #x co-ordinate of point 8
            y1 = fingerLandmarks[0][3] * cam_height #y co-ordinate of point 8       
                
            print(fingers) 
            
            cv2.rectangle(frame, (detectionArea, detectionArea), (cam_width - detectionArea, cam_height - detectionArea)
                              , (255, 0, 255), 2) 
            
            if fingers==[1,1,0,0,0]: #L sign up
                x2 = np.interp(x1, (detectionArea, cam_width - detectionArea), (0, screen_width))
                y2 = np.interp(y1, (detectionArea, cam_height - detectionArea), (0, screen_height))
                
                currentX = previousX +(x2 - previousX) / smoothening
                currentY = previousY +(y2 - previousY) / smoothening

                pyautogui.FAILSAFE = False
                pyautogui.moveTo(currentX, currentY)
                
                previousX, previousY = currentX, currentY
                gesture = "L"
                
            elif fingers==[1,1,1,1,1]: #Open Palm
                pyautogui.rightClick()
                gesture = "Open Palm"
            elif fingers==[0,1,0,0,0]: #Index Up
                pyautogui.leftClick()
                gesture = "Index Up"
            elif fingers==[0,1,0,0,1]: #Horns
                pyautogui.scroll(25)
                gesture = "Horns"
            elif fingers==[1,1,0,0,1]: #Spiderman
                pyautogui.scroll(-25)    
                gesture = "Spiderman"
            elif fingers==[0,0,0,0,0]: #fist
                with pyautogui.hold('alt'):
                    pyautogui.press('f4')
                gesture = "Fist"   
    
    cTime = time.time()
    fps = str(int(1/ (cTime - pTime)))
    pTime = cTime
    
    cv2.putText(frame, "FPS: " + fps, (20, 60), font, fontScale,  
                 color, thickness, cv2.LINE_AA, False)
 
    cv2.putText(frame, "Gesture:" + gesture, (20, 100), font, fontScale,  
                 color, thickness, cv2.LINE_AA, False)
    
    cv2.imshow('Handsfree Mouse', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()