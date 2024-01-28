# import the opencv library 
import cv2 
  
  
# define a video capture object 
vid = cv2.VideoCapture(4) 
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('your_video.avi', fourcc, 20.0, size)
while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
    if ret==True:
        #frame = cv2.flip(frame,0)

        # write the flipped frame
        #out.write(frame)
  
    # Display the resulting frame 
        cv2.imshow('frame', frame) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    else:
        break
  
# After the loop release the cap object 
vid.release() 
#out.release()
# Destroy all the windows 
cv2.destroyAllWindows() 