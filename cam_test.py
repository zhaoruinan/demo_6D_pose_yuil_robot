# import the opencv library 
import cv2

vid = cv2.VideoCapture(0) 
while(True): 
    ret, frame = vid.read() 
    if ret==True:
        cv2.imshow('RGB', frame) 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    else:
        break
  
vid.release() 
cv2.destroyAllWindows() 