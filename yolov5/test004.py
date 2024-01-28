import numpy as np
import cv2
import imagezmq
import json
import time
import torch
import platform
import sys
sys.path.append("/home/zhuzhuxia/code/pv_net_docker/yolov5/")
from models.common import DetectMultiBackend
from utils.torch_utils import select_device, smart_inference_mode
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import masks2segments, process_mask, process_mask_native
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer)

import random
def remove_random_element(tensor):
    if len(tensor) > 0:
        index = torch.randint(0, len(tensor), (1,))
        removed_element = tensor[index]
        tensor = torch.cat((tensor[:index], tensor[index+1:]), dim=0)
        return removed_element, tensor 
    else:
        return None

dnn=False
weights = "./yolov5/runs/train-seg/exp6/weights/best.pt"
data = "./yolov5/data/screw_seg.yaml"
half=False
augment=False
view_img=True
retina_masks=True
conf_thres=0.85
iou_thres=0.9
max_det=10
line_thickness=3
agnostic_nms=False
classes=None
device = select_device(0)
class yolo_processor():
    def __init__(self):
        self.model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, self.names, pt = self.model.stride, self.model.names, self.model.pt
#results = model([im1], size=640)
#results.show()  
#image_hub = imagezmq.ImageHub(open_port='tcp://192.168.1.3:5554')
#sender = imagezmq.ImageSender(connect_to='tcp://192.168.1.3:5555')
        self.corner_2d_pred = np.array([[364, 336],
         [379, 299],
         [378, 324],
         [392, 289],
         [461, 385],
         [479, 347],
         [470, 370],
         [487, 334]])
        self.pose_pred = np.array([[ 0.86538923,  0.37343903,  0.33413286,  0.02735332],
         [ 0.40679352, -0.13418061, -0.90361197,  0.03367478],
         [-0.29260983,  0.91789915, -0.26803104,  0.32564679]])
        self.model = torch.hub.load("ultralytics/yolov5", "custom", "yolov5/runs/train-seg/exp6/weights/best.pt")  # load from PyTorch Hub (WARNING: inference not yet supported)
        seen, windows, self.dt = 0, [], (Profile(), Profile(), Profile())
    def process_yolo(self, image_name, image):
#while True:
    #image_name, image = image_hub.recv_image() 
        demo_image = image.astype(np.float32)
        im_out = image


    #im = torch.from_numpy(image).to(model.device)
    #pred, proto = model(im, augment=augment)[:2]
    #results = model([image], size=640)
    #cv2.imshow("RGB",image)
    #cv2.waitKey(1)
    #time.sleep(0.01)
    #json_load[""]
        with self.dt[0]:
            im = torch.from_numpy(image).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            #print(im.shape)
            im = torch.permute(im, (2, 0, 1))
            im = torch.unsqueeze(im, 0)
            #print(im.shape)

        # Inference
        with self.dt[1]:
            visualize =  False
            #print(im.shape)            
            pred, proto = self.model(im, augment=augment, visualize=visualize)[:2]

        # NMS
        with self.dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        im0 = image
        for i, det in enumerate(pred):  # per image
            annotator = Annotator(im0, line_width=line_thickness, example=str(self.names))
            if len(det):
                if retina_masks:
                    # scale bbox first the crop masks
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                    masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
                else:
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class

                # Mask plotting
                #print(det[:, 5])
                #print(masks.shape)
                
                #print(masks.shape)
                #det = remove_random_element(det)
                if len(det) > 1:
                    det_t = det[:, 5][:-1]
                    one_mask, masks = remove_random_element(masks)
                    
                else:
                    det_t = det[:, 5]
                    one_mask = masks[0]
                print(one_mask.shape)
                print(im0.shape)
                one_mask = np.uint8(torch.squeeze(one_mask).detach().cpu().numpy())
                im_out = cv2.bitwise_and(im0,im0, mask = one_mask)
                im_out[np.where(one_mask ==0)]=[255, 255, 255]
                
                print(im_out.shape)
                annotator.masks(
                    masks,
                    colors=[colors(x, True) for x in det_t],
                    im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous() /
                    255 if retina_masks else im[i])

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.namedWindow("seg", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow("seg", im0.shape[1], im0.shape[0])
                im0 = cv2.cvtColor(np.array(im0), cv2.COLOR_RGB2BGR)
                cv2.imshow("seg", im0)
                if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                    exit()
        json_load = {'corner_2d_pred': self.corner_2d_pred.tolist(), 'pose_pred': self.pose_pred.tolist()}
        #text1 = sender.send_image('RealSense', im_out)
        #json_load = json.loads(text1)
        #image_hub.send_reply(json.dumps(json_load).encode("utf-8"))
        return im_out, json_load #json.dumps(json_load).encode("utf-8")
