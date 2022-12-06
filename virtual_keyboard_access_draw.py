#!/usr/bin/env python
# coding: utf-8

# ### Live Camera Feed

# In[ ]:


# Import
import cv2

# Access camera
cap = cv2.VideoCapture(0)

# Read/Show frame's from camera
while True:
    _, frame = cap.read()
    
    cv2.imshow('Live', frame)
    if cv2.waitKey(1) == 27: # ESC
        break

cap.release()


# In[16]:


import cv2
cv2.destroyAllWindows()


# In[ ]:


# !pip install mediapipe
# For M1 Chip
#!pip install mediapipe-silicon


# In[ ]:


import mediapipe as mp
import cv2
import numpy as np
# Load Model
hands = mp.solutions.hands
hand_landmark = hands.Hands(max_num_hands=1)
draw = mp.solutions.drawing_utils


# In[ ]:


# Import
import cv2

# Access camera
cap = cv2.VideoCapture(0)
draw = mp.solutions.drawing_utils

# Read/Show frame's from camera
while True:
    _, frame = cap.read() # BGR -> RGB
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    op = hand_landmark.process(rgb)
    
    if op.multi_hand_landmarks:
        for all_landmarks in op.multi_hand_landmarks: # 
            draw.draw_landmarks(frame, all_landmarks, hands.HAND_CONNECTIONS)
    
    cv2.imshow('Live', frame)
    if cv2.waitKey(1) == 27: # ESC
        break

cap.release()


# ### Hand and Face Tracking

# In[ ]:


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5)

draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1) # 0 vertically x-axis, 1 horizontally y-axis
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    op = hand_landmark.process(rgb)
    results = face_mesh.process(rgb)
    
    if op.multi_hand_landmarks:
        all_landmarks = next(iter(op.multi_hand_landmarks))
        draw.draw_landmarks(frame, all_landmarks, hands.HAND_CONNECTIONS)
        
    if results.multi_face_landmarks:
        for i in results.multi_face_landmarks:
            draw.draw_landmarks(frame, i, mp_face_mesh.FACEMESH_IRISES)
    
    cv2.imshow('Live', frame)
    
    if cv2.waitKey(10) == 27:
        break
  
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)


# # Draw
# 
# 1. Locate index finger
# 2. draw line

# In[ ]:


# Camera frame resolution
frame_shape = frame.shape


# In[ ]:


# Import
import cv2
import numpy as np

prevxy = None
mask = np.zeros(frame_shape, dtype='uint8') # to premanently draw
colour = (123, 34, 90)
thickness = 4

# Access camera
cap = cv2.VideoCapture(0)
draw = mp.solutions.drawing_utils

# Read/Show frame's from camera
while True:
    _, frame = cap.read() # BGR -> RGB
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    op = hand_landmark.process(rgb)
    
    if op.multi_hand_landmarks:
        for all_landmarks in op.multi_hand_landmarks: # 
            draw.draw_landmarks(frame, all_landmarks, hands.HAND_CONNECTIONS)
            
            x = int(all_landmarks.landmark[8].x*frame_shape[1])
            y = int(all_landmarks.landmark[8].y*frame_shape[0])
            
            if prevxy != None:
                # draw stuf
                cv2.line(mask, prevxy, (x, y), colour, thickness)
            prevxy = (x, y)
            
    # Merge Frame and Mask
    frame = np.where(mask, mask, frame)
    
    cv2.imshow('Live', frame)
    if cv2.waitKey(1) == 27: # ESC
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)


# # Eraser

# In[ ]:


# Import
import cv2
import numpy as np

prevxy = None
# mask = np.zeros(frame_shape, dtype='uint8') # to premanently draw, 0 values are invisible
colour = (123, 34, 90)
thickness = 4

# Access camera
cap = cv2.VideoCapture(0)
draw = mp.solutions.drawing_utils

# Read/Show frame's from camera
while True:
    _, frame = cap.read() # BGR -> RGB
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    op = hand_landmark.process(rgb)
    
    if op.multi_hand_landmarks:
        for all_landmarks in op.multi_hand_landmarks: # 
            draw.draw_landmarks(frame, all_landmarks, hands.HAND_CONNECTIONS)
            
            # Index Finger Location
            x = int(all_landmarks.landmark[8].x*frame_shape[1])
            y = int(all_landmarks.landmark[8].y*frame_shape[0])
            
            cv2.circle(frame, (x, y), 30, (0,0,0), -1) # -1 means fill
            cv2.circle(mask, (x, y), 30, (0,0,0), -1) # -1 means fill
            
    # Merge Frame and Mask
    frame = np.where(mask, mask, frame)
    
    cv2.imshow('Live', frame)
    if cv2.waitKey(1) == 27: # ESC
        break

cap.release()


# In[ ]:




