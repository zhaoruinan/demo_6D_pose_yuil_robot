from lib.config import cfg, args
import numpy as np
from numpy import polyfit,poly1d
import os
import math
pi = math.pi
from math import cos,sin
def run_rgb():
    import glob
    from scipy.misc import imread
    import matplotlib.pyplot as plt

    syn_ids = sorted(os.listdir('data/ShapeNet/renders/02958343/'))[-10:]
    for syn_id in syn_ids:
        pkl_paths = glob.glob('data/ShapeNet/renders/02958343/{}/*.pkl'.format(syn_id))
        np.random.shuffle(pkl_paths)
        for pkl_path in pkl_paths:
            img_path = pkl_path.replace('_RT.pkl', '.png')
            img = imread(img_path)
            plt.imshow(img)
            plt.show()


def run_dataset():
    from lib.datasets import make_data_loader
    import tqdm

    cfg.train.num_workers = 0
    data_loader = make_data_loader(cfg, is_train=False)
    for batch in tqdm.tqdm(data_loader):
        pass


def run_network():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    import time

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    total_time = 0
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            network(batch['inp'], batch)
            torch.cuda.synchronize()
            total_time += time.time() - start
    print(total_time / len(data_loader))


def run_evaluate():
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils.net_utils import load_network

    torch.manual_seed(0)

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    for batch in tqdm.tqdm(data_loader):
        inp = batch['inp'].cuda()
        with torch.no_grad():
            output = network(inp)
        evaluator.evaluate(output, batch)
    evaluator.summarize()


def run_visualize():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)
        visualizer.visualize(output, batch)


def run_visualize_train():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=True)
    visualizer = make_visualizer(cfg, 'train')
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)
        visualizer.visualize_train(output, batch)


def run_analyze():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.analyzers import make_analyzer

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    cfg.train.num_workers = 0
    data_loader = make_data_loader(cfg, is_train=False)
    analyzer = make_analyzer(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)
        analyzer.analyze(output, batch)


def run_net_utils():
    from lib.utils import net_utils
    import torch
    import os

    model_path = 'data/model/rcnn_snake/rcnn/139.pth'
    pretrained_model = torch.load(model_path)
    net = pretrained_model['net']
    net = net_utils.remove_net_prefix(net, 'dla.')
    net = net_utils.remove_net_prefix(net, 'cp.')
    pretrained_model['net'] = net
    model_path = 'data/model/rcnn_snake/rcnn/139.pth'
    os.system('mkdir -p {}'.format(os.path.dirname(model_path)))
    torch.save(pretrained_model, model_path)


def run_linemod():
    from lib.datasets.linemod import linemod_to_coco
    linemod_to_coco.linemod_to_coco(cfg)


def run_tless():
    from lib.datasets.tless import handle_rendering_data, fuse, handle_test_data, handle_ag_data, tless_to_coco
    # handle_rendering_data.render()
    # handle_rendering_data.render_to_coco()
    # handle_rendering_data.prepare_asset()

    # fuse.fuse()
    # handle_test_data.get_mask()
    # handle_test_data.test_to_coco()
    handle_test_data.test_pose_to_coco()

    # handle_ag_data.ag_to_coco()
    # handle_ag_data.get_ag_mask()
    # handle_ag_data.prepare_asset()

    # tless_to_coco.handle_train_symmetry_pose()
    # tless_to_coco.tless_train_to_coco()


def run_ycb():
    from lib.datasets.ycb import handle_ycb
    handle_ycb.collect_ycb()


def run_render():
    from lib.utils.renderer import opengl_utils
    from lib.utils.vsd import inout
    from lib.utils.linemod import linemod_config
    import matplotlib.pyplot as plt

    obj_path = 'data/linemod/cat/cat.ply'
    model = inout.load_ply(obj_path)
    model['pts'] = model['pts'] * 1000.
    im_size = (640, 300)
    opengl = opengl_utils.NormalRender(model, im_size)

    K = linemod_config.linemod_K
    pose = np.load('data/linemod/cat/pose/pose0.npy')
    depth = opengl.render(im_size, 100, 10000, K, pose[:, :3], pose[:, 3:] * 1000)

    plt.imshow(depth)
    plt.show()


def run_custom():
    from tools import handle_custom_dataset
    data_root = 'data/custom'
    handle_custom_dataset.sample_fps_points(data_root)
    handle_custom_dataset.custom_to_coco(data_root)


def run_detector_pvnet():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer

    network = make_network(cfg).cuda()
    print(cfg.model_dir)
    print(cfg.test.epoch)
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)
        visualizer.visualize(output, batch)
def visualize_ (output):

        from lib.utils.pvnet import pvnet_pose_utils
        from lib.utils import img_utils
        #mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        #inp = img_utils.unnormalize_img(inp, mean, std).permute(1, 2, 0)
        #print(inp)
        kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()
        #print(kpt_2d)
     #   img_id = int(batch['img_id'][0])
    #    anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        #K = np.array([[774.66809765 ,  0. ,313.05165248],
#[  0. ,775.23750177,211.69164865],
#[  0.  , 0. ,  1.]])
        K = np.array([[800.9824513   ,  0. ,321.68465049],
[  0. ,800.20088847,230.59126429],
[  0.  , 0. ,  1.]])
#        K = np.array([[735.320751   ,  0. ,230.59126429],
#[  0. ,732.90725684,272.73279961],
#[  0.  , 0. ,  1.]])
#        K = np.array([[605.28 ,  0. ,325.73],
# [  0. ,603.868 ,236.881],
# [  0.  , 0. ,  1.]])
#        K = np.array([[758.377 ,  0. ,319.42],
# [  0. ,758.59 ,226.155],
# [  0.  , 0. ,  1.]])
#        K = np.array([[381.74 ,  0. ,318.055],
#         [  0. ,381.74 ,235.08],
#         [  0.  , 0. ,  1.]])
        #fps_3d = np.array([[4.442409135663183e-06, -0.007305943872779608, 0.006071927957236767], [0.04799924045801163, 0.004776963964104652, 0.0018356989603489637], [0.00589775713160634, 0.0054193842224776745, -0.009460913017392159], [0.007862349972128868, 0.007024947088211775, 0.006832438055425882], [0.03779765963554382, -0.00533895893022418, -0.0024005300365388393], [0.011210310272872448, -0.00533895893022418, -0.0021181150805205107], [0.00018627849931363016, -0.005619957111775875, -0.008048836141824722], [0.03464236110448837, 0.005619957111775875, 0.0018356989603489637]])
        fps_3d = np.array([[-2.467565983533859253e-02, -4.346641898155212402e-02, -1.404704991728067398e-02],
[-2.490215934813022614e-02, 4.253591969609260559e-02, 1.541784964501857758e-02],
[2.315844967961311340e-02, -4.204858094453811646e-02, 1.609643921256065369e-02],
[2.216614037752151489e-02, 4.273251071572303772e-02, -1.433391962200403214e-02],
[1.070148032158613205e-02, -3.149121999740600586e-02, -1.772985048592090607e-02],
[-2.373323962092399597e-02, -2.307748049497604370e-02, 1.691528968513011932e-02],
[-1.204913016408681870e-02, 3.013489022850990295e-02, -1.610564999282360077e-02],
[2.352176047861576080e-02, 2.225551940500736237e-02, 1.459936983883380890e-02]])
        center_3d = np.array([0.0, 0.0, 0.0])
        kpt_3d = np.concatenate([fps_3d, [center_3d]], axis=0)
        
        pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)
        corner_3d = np.array([[-0.02624396, -0.04361221, -0.01792065],
 [-0.02624396, -0.04361221,  0.01792066],
 [-0.02624396,  0.04361221, -0.01792065],
 [-0.02624396,  0.04361221,  0.01792066],
 [ 0.02624396, -0.04361221, -0.01792065],
 [ 0.02624396, -0.04361221,  0.01792066],
 [ 0.02624396,  0.04361221, -0.01792065],
 [ 0.02624396,  0.04361221,  0.01792066]])
        #corner_3d = np.array(anno['corner_3d'])
        #corner_2d_gt = pvnet_pose_utils.project(corner_3d, K, pose_gt)
        corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred)
        return corner_2d_pred, pose_pred
        #_, ax = plt.subplots(1)
        #ax.imshow(inp)
        #ax.add_patch(patches.Polygon(xy=corner_2d_gt[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='g'))
        #ax.add_patch(patches.Polygon(xy=corner_2d_gt[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='g'))
        #ax.add_patch(patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
        #ax.add_patch(patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))
        #plt.show()
def run_demo():
    from lib.datasets import make_data_loader
    from lib.visualizers import make_visualizer
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils.net_utils import load_network
    import glob
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import cv2
    #from yolov5.test005 import yolo_processor as yolo 
    import sys
    #yolo_worker = yolo()
    torch.manual_seed(0)
    meta = np.load(os.path.join(cfg.demo_path, 'meta.npy'), allow_pickle=True).item()
    cap = cv2.VideoCapture("001.mp4") 
    network = make_network(cfg).cuda()
    print(cfg.model_dir)
    print(cfg.test.epoch)
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    visualizer = make_visualizer(cfg)

    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    
    while (cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480), 
               interpolation = cv2.INTER_LINEAR)
        #im_out, json_dumps=yolo_worker.process_yolo('RealSense', frame)
        demo_image = frame
        inp = (((demo_image/255.)-mean)/std).transpose(2, 0, 1).astype(np.float32)
        inp = torch.Tensor(inp[None]).cuda()
        with torch.no_grad():
            output = network(inp)
        corner_2d_pred=visualize_(output)
        corner_2d_pred = np.int32(corner_2d_pred)
        points1 = np.array([corner_2d_pred[5],corner_2d_pred[4],corner_2d_pred[6],corner_2d_pred[7],corner_2d_pred[5],corner_2d_pred[1],corner_2d_pred[3],corner_2d_pred[7]])
        points2 = np.array([corner_2d_pred[0],corner_2d_pred[1],corner_2d_pred[3],corner_2d_pred[2],corner_2d_pred[0],corner_2d_pred[4],corner_2d_pred[6],corner_2d_pred[2]])
        demo_image_ = cv2.cvtColor(np.array(demo_image_), cv2.COLOR_RGB2BGR)
        demo_image_ = np.uint8(demo_image_)
        try:
            cv2.polylines(demo_image_, [points1], True, (255, 0, 0), thickness=1)
            cv2.polylines(demo_image_, [points2], True, (255, 0, 0), thickness=1)            
        except:
            pass
        cv2.imshow("RGB",demo_image_)
        cv2.waitKey(1)
        import time
        time.sleep(0.01)


def run_online():
    from lib.datasets import make_data_loader
    from lib.visualizers import make_visualizer
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils.net_utils import load_network
    import glob
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import cv2
    import imagezmq
    import json
    image_hub = imagezmq.ImageHub(open_port='tcp://192.168.1.3:5555')
    torch.manual_seed(0)
    print(os.path.join(cfg.demo_path))
    meta = np.load(os.path.join(cfg.demo_path, 'meta.npy'), allow_pickle=True).item()
    demo_images = glob.glob(cfg.demo_path + '/*jpg')
    network = make_network(cfg).cuda()
    print(cfg.model_dir)
    print(cfg.test.epoch)
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    visualizer = make_visualizer(cfg)

    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    
    while True:
        image_name, image = image_hub.recv_image()  
        demo_image_= image
        demo_image = image.astype(np.float32)
        inp = (((demo_image/255.)-mean)/std).transpose(2, 0, 1).astype(np.float32)
        inp = torch.Tensor(inp[None]).cuda()
        with torch.no_grad():
            output = network(inp)
        corner_2d_pred,pose_pred =visualize_(output)
        corner_2d_pred = np.int32(corner_2d_pred)
        points1 = np.array([corner_2d_pred[5],corner_2d_pred[4],corner_2d_pred[6],corner_2d_pred[7],corner_2d_pred[5],corner_2d_pred[1],corner_2d_pred[3],corner_2d_pred[7]])
        points2 = np.array([corner_2d_pred[0],corner_2d_pred[1],corner_2d_pred[3],corner_2d_pred[2],corner_2d_pred[0],corner_2d_pred[4],corner_2d_pred[6],corner_2d_pred[2]])
        #demo_image_ = cv2.cvtColor(np.array(demo_image_), cv2.COLOR_RGB2BGR)
        demo_image_ = np.uint8(demo_image_)
        try:
            cv2.polylines(demo_image_, [points1], True, (255, 0, 0), thickness=1)
            cv2.polylines(demo_image_, [points2], True, (255, 0, 0), thickness=1)            
        except:
            pass
        print("corner_2d_pred",corner_2d_pred)
        print("pose_pred",pose_pred)
        ack ={'corner_2d_pred': corner_2d_pred.tolist(), 'pose_pred':pose_pred.tolist()}
        image_hub.send_reply(json.dumps(ack).encode("utf-8"))
        #cv2.namedWindow("seg", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        #cv2.resizeWindow("seg", demo_image_.shape[1], demo_image_.shape[0])
        cv2.imshow("RGB",demo_image_)
        cv2.waitKey(1)
        import time
        time.sleep(0.01)
def run_online2():
    from lib.datasets import make_data_loader
    from lib.visualizers import make_visualizer
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils.net_utils import load_network
    import glob
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import cv2
    from yolov5.test004 import yolo_processor as yolo 
    import sys
    yolo_worker = yolo()
    vid = cv2.VideoCapture(0) 
    #vid = cv2.VideoCapture("001.mp4") 
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    vid.set(cv2.CAP_PROP_FPS, 30)
    #width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    #height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    width,height = 720,480
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    torch.manual_seed(0)
    print(os.path.join(cfg.demo_path))
    #meta = np.load(os.path.join(cfg.demo_path, 'meta.npy'), allow_pickle=True).item()
    #demo_images = glob.glob(cfg.demo_path + '/*jpg')
    network = make_network(cfg).cuda()
    print(cfg.model_dir)
    print(cfg.test.epoch)
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    visualizer = make_visualizer(cfg)

    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
#out = cv2.VideoWriter('your_video.avi', fourcc, 20.0, size)


    while(True): 
      
    # Capture the video frame 
    # by frame 
        ret, frame = vid.read() 
        if ret==True:

            print(frame.shape)
            frame = cv2.resize(frame, (640, 480), 
               interpolation = cv2.INTER_LINEAR)
            im_out, json_dumps=yolo_worker.process_yolo('RealSense', frame)
            #cv2.imshow('im_out', im_out) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

            demo_image_= im_out
            demo_image = im_out.astype(np.float32)
            inp = (((demo_image/255.)-mean)/std).transpose(2, 0, 1).astype(np.float32)
            inp = torch.Tensor(inp[None]).cuda()
            with torch.no_grad():
                output = network(inp)
            corner_2d_pred,pose_pred =visualize_(output)
            corner_2d_pred = np.int32(corner_2d_pred)
            points1 = np.array([corner_2d_pred[5],corner_2d_pred[4],corner_2d_pred[6],corner_2d_pred[7],corner_2d_pred[5],corner_2d_pred[1],corner_2d_pred[3],corner_2d_pred[7]])
            points2 = np.array([corner_2d_pred[0],corner_2d_pred[1],corner_2d_pred[3],corner_2d_pred[2],corner_2d_pred[0],corner_2d_pred[4],corner_2d_pred[6],corner_2d_pred[2]])
            #demo_image_ = cv2.cvtColor(np.array(demo_image_), cv2.COLOR_RGB2BGR)
            demo_image_ = np.uint8(demo_image_)
            try:
                cv2.polylines(demo_image_, [points1], True, (255, 0, 0), thickness=1)
                cv2.polylines(demo_image_, [points2], True, (255, 0, 0), thickness=1)            
            except:
               pass
            #print("corner_2d_pred",corner_2d_pred)
            #print("pose_pred",pose_pred)
            ack ={'corner_2d_pred': corner_2d_pred.tolist(), 'pose_pred':pose_pred.tolist()}
            cv2.namedWindow("seg", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow("seg", demo_image_.shape[1], demo_image_.shape[0])
            image_show = np.hstack((frame,demo_image_))
            cv2.imshow("RGB",image_show)
            cv2.waitKey(1)
            import time
            time.sleep(0.01)

# After the loop release the cap object 
    vid.release() 
#out.release()
# Destroy all the windows 
    cv2.destroyAllWindows() 

def run_online3():
    from lib.datasets import make_data_loader
    from lib.visualizers import make_visualizer
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils.net_utils import load_network
    import glob
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import cv2
    import imagezmq
    import json
    vid = cv2.VideoCapture(2) 
    #vid = cv2.VideoCapture("001.mp4") 
    #vid = cv2.VideoCapture("assets/Webcam/006.webm") 
    torch.manual_seed(0)
    print(os.path.join(cfg.demo_path))
    network = make_network(cfg).cuda()
    print(cfg.model_dir)
    print(cfg.test.epoch)
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    visualizer = make_visualizer(cfg)

    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    import time
    while True:
        start_time = time.time()
        ret, frame = vid.read() 
        frame = cv2.resize(frame, (640, 480), 
               interpolation = cv2.INTER_LINEAR)
        demo_image_= frame
        demo_image = frame.astype(np.float32)
        inp = (((demo_image/255.)-mean)/std).transpose(2, 0, 1).astype(np.float32)
        #print(inp)
        inp = torch.Tensor(inp[None]).cuda()
        with torch.no_grad():
            output = network(inp)
        #print(output)
        corner_2d_pred,pose_pred =visualize_(output)
        corner_2d_pred = np.int32(corner_2d_pred)
        points1 = np.array([corner_2d_pred[5],corner_2d_pred[4],corner_2d_pred[6],corner_2d_pred[7],corner_2d_pred[5],corner_2d_pred[1],corner_2d_pred[3],corner_2d_pred[7]])
        points2 = np.array([corner_2d_pred[0],corner_2d_pred[1],corner_2d_pred[3],corner_2d_pred[2],corner_2d_pred[0],corner_2d_pred[4],corner_2d_pred[6],corner_2d_pred[2]])
        #demo_image_ = cv2.cvtColor(np.array(demo_image_), cv2.COLOR_RGB2BGR)
        demo_image_ = np.uint8(demo_image_)
        try:
            cv2.polylines(demo_image_, [points1], True, (255, 0, 0), thickness=1)
            cv2.polylines(demo_image_, [points2], True, (255, 0, 0), thickness=1)            
        except:
            print("passsssssssssssssssssssssssssssssssss")
            pass
        #print("corner_2d_pred",corner_2d_pred)
        #print("pose_pred",pose_pred)
        ack ={'corner_2d_pred': corner_2d_pred.tolist(), 'pose_pred':pose_pred.tolist()}

        cv2.namedWindow("seg", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow("seg", demo_image_.shape[1], demo_image_.shape[0])
        cv2.imshow("RGB",demo_image_)
        cv2.waitKey(1)
        print("--- %s seconds ---" % (time.time() - start_time))
        #import time
        #time.sleep(0.01)
def run_demo3():
    from lib.datasets import make_data_loader
    from lib.visualizers import make_visualizer
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils.net_utils import load_network
    import glob
    from PIL import Image
    import cv2

    torch.manual_seed(0)
    demo_images = glob.glob(cfg.demo_path + '/*jpg')

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    visualizer = make_visualizer(cfg)

    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    for demo_image in demo_images:
        demo_image = np.array(Image.open(demo_image)).astype(np.float32)
        demo_image = cv2.resize(demo_image, (640, 480), 
               interpolation = cv2.INTER_LINEAR)
        demo_image_ = demo_image.copy()
        inp = (((demo_image/255.)-mean)/std).transpose(2, 0, 1).astype(np.float32)
        #print(inp)
        inp = torch.Tensor(inp[None]).cuda()
        with torch.no_grad():
            output = network(inp)
        #print(output)
        corner_2d_pred,pose_pred =visualize_(output)
        corner_2d_pred = np.int32(corner_2d_pred)
        points1 = np.array([corner_2d_pred[5],corner_2d_pred[4],corner_2d_pred[6],corner_2d_pred[7],corner_2d_pred[5],corner_2d_pred[1],corner_2d_pred[3],corner_2d_pred[7]])
        points2 = np.array([corner_2d_pred[0],corner_2d_pred[1],corner_2d_pred[3],corner_2d_pred[2],corner_2d_pred[0],corner_2d_pred[4],corner_2d_pred[6],corner_2d_pred[2]])
        demo_image_ = cv2.cvtColor(np.array(demo_image_), cv2.COLOR_RGB2BGR)
        demo_image_ = np.uint8(demo_image_)
        try:
            cv2.polylines(demo_image_, [points1], True, (255, 0, 0), thickness=1)
            cv2.polylines(demo_image_, [points2], True, (255, 0, 0), thickness=1)            
        except:
            pass
        print("corner_2d_pred",corner_2d_pred)
        print("pose_pred",pose_pred)
        ack ={'corner_2d_pred': corner_2d_pred.tolist(), 'pose_pred':pose_pred.tolist()}

        #cv2.namedWindow("seg", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        #cv2.resizeWindow("seg", demo_image_.shape[1], demo_image_.shape[0])
        cv2.imshow("RGB",demo_image_)
        cv2.waitKey(1)
        #import time
        #time.sleep(0.01)

import socket
HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 8089  # The port used by the server
class real_robot(object):
    def __init__(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    def robot_read_pos(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(b"READ")
            data = s.recv(1024)
            pose_t = np.frombuffer(data, dtype=np.float64)
        return pose_t
        #print(f"Received {data!r}")
    def robot_set_pos(self,pose = None,home = False,stop = False,axis = 1,read = False):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            if read:
                s.sendall(b"READ")
                data = s.recv(1024)
                pose_t = np.frombuffer(data, dtype=np.float64)
                return pose_t
            elif home:
                s.sendall(b"HOME")
            elif stop:
                s.sendall(b"STOP")
            else:
                pose_b = pose.tobytes()
            if axis ==1:
                s.sendall(b"PS_T"+pose_b)
            else:
                s.sendall(b"PS_J"+pose_b)
            data = s.recv(1024)
            pose_j = np.frombuffer(data, dtype=np.float64)
            return pose_j
        #print(f"Received {data!r}")
        #print(pose_j)
def TransToRp(T):
    """Converts a homogeneous transformation matrix into a rotation matrix
    and position vector
    :param T: A homogeneous transformation matrix
    :return R: The corresponding rotation matrix,
    :return p: The corresponding position vector.
    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        (np.array([[1, 0,  0],
                   [0, 0, -1],
                   [0, 1,  0]]),
         np.array([0, 0, 3]))
    """
    T = np.array(T)
    return T[0: 3, 0: 3], T[0: 3, 3]
def RpToTrans(R, p):
    """Converts a rotation matrix and a position vector into homogeneous
    transformation matrix
    :param R: A 3x3 rotation matrix
    :param p: A 3-vector
    :return: A homogeneous transformation matrix corresponding to the inputs
    Example Input:
        R = np.array([[1, 0,  0],
                      [0, 0, -1],
                      [0, 1,  0]])
        p = np.array([1, 2, 5])
    Output:
        np.array([[1, 0,  0, 1],
                  [0, 0, -1, 2],
                  [0, 1,  0, 5],
                  [0, 0,  0, 1]])
    """
    return np.r_[np.c_[R, p], [[0, 0, 0, 1]]]

def angle2rotation(x, y, z):
    Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
    Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
    Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    return R
def gripper2base(x, y, z, tx, ty, tz):
    thetaX = x #/ 180 * pi
    thetaY = y #/ 180 * pi
    thetaZ = z #/ 180 * pi
    R_gripper2base = angle2rotation(thetaX, thetaY, thetaZ)
    T_gripper2base = np.array([[tx], [ty], [tz]])
    Matrix_gripper2base = np.column_stack([R_gripper2base, T_gripper2base])
    Matrix_gripper2base = np.row_stack((Matrix_gripper2base, np.array([0, 0, 0, 1])))
    R_gripper2base = Matrix_gripper2base[:3, :3]
    T_gripper2base = Matrix_gripper2base[:3, 3].reshape((3, 1))
    return R_gripper2base, T_gripper2base
def rotm2euler(R) :
 
    assert(isRotm(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])
def isRotm(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
def run_online4():
    test_time = "002"
    path_list = ["pose_data/RGB/", "pose_data/seg/", "pose_data/pose_pred/","pose_data/time/"]
    for path in path_list:
        if not os.path.exists(path):
            os.mkdir(path)
    for path in path_list:
        if not os.path.exists(path+test_time+'/'):
            os.mkdir(path+test_time+'/')
    import threading
    #from menu import menu
    #t_menu = threading.Thread(target=menu)
    #t_menu.start()
    pose_obs = np.array([0.489, 0.0981, 0.413, 2.589, 0.245, 1.919])
    #robot_set_pos(home = True)
    pose_move = pose_obs
    robot1 = real_robot()
    robot1.robot_set_pos(pose = pose_move)

    from lib.datasets import make_data_loader
    from lib.visualizers import make_visualizer
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils.net_utils import load_network
    import glob
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import cv2
    from yolov5.test004 import yolo_processor as yolo 
    import sys
    yolo_worker = yolo()
    #vid = cv2.VideoCapture("test003.webm") 
    vid = cv2.VideoCapture(0) 
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    vid.set(cv2.CAP_PROP_FPS, 30)
    #width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    #height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    width,height = 720,480
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    torch.manual_seed(0)
    print(os.path.join(cfg.demo_path))
    #meta = np.load(os.path.join(cfg.demo_path, 'meta.npy'), allow_pickle=True).item()
    #demo_images = glob.glob(cfg.demo_path + '/*jpg')
    network = make_network(cfg).cuda()
    print(cfg.model_dir)
    print(cfg.test.epoch)
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    visualizer = make_visualizer(cfg)
    run_num = 1
    import time
    start_time = time.time()

    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

    pose_path = []
    pose_path_t = []
    move_flag = False
#out = cv2.VideoWriter('your_video.avi', fourcc, 20.0, size)



    while(True): 
      
    # Capture the video frame 
    # by frame 
        ret, frame = vid.read() 
        if ret==True:

            #print(frame.shape)
            frame = cv2.resize(frame, (640, 480), 
               interpolation = cv2.INTER_LINEAR)
            frame_r = frame.copy()
            im_out, json_dumps=yolo_worker.process_yolo('RealSense', frame)
            #cv2.imshow('im_out', im_out) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

            demo_image_= im_out
            demo_image = im_out.astype(np.float32)
            inp = (((demo_image/255.)-mean)/std).transpose(2, 0, 1).astype(np.float32)
            inp = torch.Tensor(inp[None]).cuda()
            with torch.no_grad():
                output = network(inp)
            corner_2d_pred,pose_pred =visualize_(output)
            corner_2d_pred = np.int32(corner_2d_pred)

            points1 = np.array([corner_2d_pred[5],corner_2d_pred[4],corner_2d_pred[6],corner_2d_pred[7],corner_2d_pred[5],corner_2d_pred[1],corner_2d_pred[3],corner_2d_pred[7]])
            points2 = np.array([corner_2d_pred[0],corner_2d_pred[1],corner_2d_pred[3],corner_2d_pred[2],corner_2d_pred[0],corner_2d_pred[4],corner_2d_pred[6],corner_2d_pred[2]])
            #demo_image_ = cv2.cvtColor(np.array(demo_image_), cv2.COLOR_RGB2BGR)
            demo_image_ = np.uint8(demo_image_)
            try:
                cv2.polylines(demo_image_, [points1], True, (255, 0, 0), thickness=1)
                cv2.polylines(demo_image_, [points2], True, (255, 0, 0), thickness=1)            
            except:
               pass
            #print("corner_2d_pred",corner_2d_pred)
            #print("pose_pred",pose_pred)

            #xyz1 [     489.01      98.097      412.99       2.589     0.24498       1.919]


            xyz1 = robot1.robot_set_pos(read = True)
            #print(xyz1.size)
            if xyz1.size != 6:
                continue
            #print("xyz1",xyz1)
            t2b_R,t2b_T = gripper2base(xyz1[3],xyz1[4],xyz1[5]-pi/2,xyz1[0]/1000,xyz1[1]/1000,xyz1[2]/1000)
            #print(t2b_R,t2b_T)
            t2b_TR = RpToTrans(t2b_R,t2b_T)
            #print(t2b_TR)

            #c2t_TR = np.matrix([[ 0.27906511, -0.95956969,  0.03672428, 0.12282908],
            #           [ -0.94423484, -0.26724354,  0.19235762,  -0.09753307],
            #           [ -0.17476622, -0.08835665, -0.98063748, 0.04831654],
            #           [ 0.,          0.,          0.,          1.        ]])

            c2t_TR = np.matrix([[ -0.28172572, -0.95847456,  0.04423947, 0.00244391],
                       [ 0.94093182, -0.28500701 , -0.18280675,  0.09010415 ],
                       [ 0.18782418,  -0.00987504,  0.98215302, 0.0771455],
                       [ 0.,          0.,          0.,          1.        ]])

            #c2b = t2b_TR
            c2b = np.dot(t2b_TR, c2t_TR)
            #print("c2b",c2b)

            o2c_euler = rotm2euler(pose_pred[:, :3].T)
            o2c_euler = o2c_euler
            o2c_euler[0] = pi/2 - o2c_euler[0] 

            o2c_T = pose_pred[:, 3:].T[0]
            temp = o2c_T[0]
            o2c_T[0] = -o2c_T[1]
            o2c_T[1] = temp
            o2c_R = angle2rotation(o2c_euler[0],o2c_euler[1],o2c_euler[2])

            
            o2c_TR = RpToTrans(o2c_R,o2c_T)
            o2b_TR = np.dot(t2b_TR,o2c_TR)
            #print()
            o2b_R,o2b_T = TransToRp(o2b_TR) 
            move_goal_R = rotm2euler(o2b_R)
            #print("o2b_TR:",o2b_TR)
            
            t_pose = time.time() - start_time
            pose_set = pose_obs.copy()
            pose_set[0] = o2b_T[0]
            pose_set[1] = o2b_T[1]
            pose_set[2] = 0.1
            pose_set[3] = 3.14
            pose_set[4] = 0
            pose_set[5] = 1.57
            #print("pose_path and time",pose_path,pose_path_t)
            xyz_obs = np.array([     489.01,   98.097,      412.99,       2.589,     0.24498,       1.919])
            if np.linalg.norm((xyz1 - xyz_obs))<30:
                obs_flag = True
            else:
                obs_flag = False

            #print("obs_flag",obs_flag)
            if move_flag:
                #print("moving ...")
                xyz_m = np.array([xyz1[0]/1000,xyz1[1]/1000,xyz1[2]/1000])
                #print(pose_move[:3])
                distance = np.linalg.norm((xyz_m - pose_move[:3]))
                #print(distance)
                if distance <0.01:
                    
                    print("reach and go back")
                    move_flag = False
                    print("moved in --- %s seconds ---" % (time.time() - move_start))
                    time.sleep(5)
                    pose_move = pose_obs.copy()
                    print("pose_move",pose_move)
                    robot1.robot_set_pos(pose = pose_move)
                    move_start = time.time()



            robot_set_pos_np= np.array(pose_path)
            if robot_set_pos_np.size>119 and move_flag == False:
                print("--- %s seconds ---" % (time.time() - start_time))
                #print("pose_path and time",pose_path,pose_path_t)
                pose_path_t_np= np.array(pose_path_t)
                pose_path = []
                pose_path_t = []
                
                
                xy_fit = xy_line_fit(robot_set_pos_np,pose_path_t_np,27)
                print("xy_fit",xy_fit)
                #if xy_fit[0]>0.8 or xy_fit[1]>0.8 or xy_fit[0]<0.2 or xy_fit[1]<-0.8:
                #    time.sleep(3)
                #else:
                #pose_move =  np.mean(robot_set_pos_np,axis=0).copy()
                #print(pose_move)
                if move_flag ==False:
                    np.savez_compressed("pose_path" + str(int(t_pose)).zfill(5) , robot_set_pos_np)
                    
                    np.savez_compressed("pose_path_t" + str(int(t_pose)).zfill(5) , pose_path_t_np)
                
                    
                    pose_move =  np.mean(robot_set_pos_np,axis=0).copy()
                    #pose_move[0] = xy_fit[0]
                    pose_move[1] = pose_move[1]+0.18
                    robot1.robot_set_pos(pose = pose_move)
                    move_start = time.time()
                    move_flag = True
            elif (o2b_T[2]<0.2 and  o2b_T[0] > 0.4 and o2b_T[1] > -0.75 and o2b_T[0] < 0.75  ) and obs_flag and move_flag == False:
                pose_path.append(pose_set)
                pose_path_t.append(t_pose)

            
            np.savez_compressed("pose_data/time/"+test_time+"/" + str(run_num).zfill(3) , t_pose)
            np.savez_compressed("pose_data/pose_pred/"+test_time+"/" + str(run_num).zfill(3) , pose_pred)
            cv2.imwrite("pose_data/RGB/"+test_time+"/" + str(int(run_num)).zfill(3) + ".png", frame)
            cv2.imwrite("pose_data/seg/"+test_time+"/" + str(int(run_num)).zfill(3) + ".png", frame)
            run_num = run_num +1

            ack ={'corner_2d_pred': corner_2d_pred.tolist(), 'pose_pred':pose_pred.tolist()}
            cv2.namedWindow("RGB", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            
            image_show = np.hstack((frame_r,demo_image_))
            cv2.imshow("RGB",image_show)
            cv2.resizeWindow("RGB", 2400, 960)
            cv2.waitKey(1)
            import time
            time.sleep(0.01)

# After the loop release the cap object 
    vid.release() 
#out.release()
# Destroy all the windows 
    cv2.destroyAllWindows() 



def xy_line_fit(pose_line,t_past,mov_time=20):
    x_past = pose_line[:,0]
    y_past = pose_line[:,1]
    coeff_x =  polyfit(t_past, x_past,1)    
    coeff_y =  polyfit(t_past, y_past,1)    
    fx = poly1d(coeff_x)
    fy = poly1d(coeff_y)
    x = fx(t_past[-1]+mov_time)
    y = fy(t_past[-1]+mov_time)
    return [x,y]
if __name__ == '__main__':
    globals()['run_'+args.type]()

