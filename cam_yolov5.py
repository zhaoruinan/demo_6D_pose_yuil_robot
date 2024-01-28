# import the opencv library 
import cv2
import imagezmq 
from yolov5.test004 import yolo_processor as yolo 
import sys
sys.path.append("/home/yuil/code/pose_yuil_robot/yolov5/")
# define a video capture object 
yolo_worker = yolo()
vid = cv2.VideoCapture(0) 
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('your_video.avi', fourcc, 20.0, size)

sender = imagezmq.ImageSender(connect_to='tcp://192.168.1.3:5555')
while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
    if ret==True:

        im_out, json_dumps=yolo_worker.process_yolo('RealSense', frame)
        text1 = sender.send_image('RealSense', im_out)
        json_load = json.loads(text1)

        sender.send_image('RealSense', frame)
        #frame = cv2.flip(frame,0)

        # write the flipped frame
        #out.write(frame)
  
    # Display the resulting frame 
        cv2.imshow('im_out', im_out) 
      
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