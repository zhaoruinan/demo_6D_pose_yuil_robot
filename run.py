from lib.config import cfg, args
import numpy as np
import os


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
       # print(kpt_2d)
     #   img_id = int(batch['img_id'][0])
    #    anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        K = np.array([[605.28 ,  0. ,325.73],
 [  0. ,603.868 ,236.881],
 [  0.  , 0. ,  1.]])
#        K = np.array([[381.74 ,  0. ,318.055],
#         [  0. ,381.74 ,235.08],
#         [  0.  , 0. ,  1.]])
        fps_3d = np.array([[4.442409135663183e-06, -0.007305943872779608, 0.006071927957236767], [0.04799924045801163, 0.004776963964104652, 0.0018356989603489637], [0.00589775713160634, 0.0054193842224776745, -0.009460913017392159], [0.007862349972128868, 0.007024947088211775, 0.006832438055425882], [0.03779765963554382, -0.00533895893022418, -0.0024005300365388393], [0.011210310272872448, -0.00533895893022418, -0.0021181150805205107], [0.00018627849931363016, -0.005619957111775875, -0.008048836141824722], [0.03464236110448837, 0.005619957111775875, 0.0018356989603489637]])
        center_3d = np.array([0.024000072718565, 0.0, 0.0])
        kpt_3d = np.concatenate([fps_3d, [center_3d]], axis=0)
        
        pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)
        corner_3d = np.array([[ 7.543713e-08 ,-1.088813e-02 ,-9.500001e-03],
 [ 7.543713e-08 ,-1.088813e-02 , 9.500001e-03],
 [ 7.543713e-08 , 1.088813e-02,-9.500001e-03],
 [ 7.543713e-08 , 1.088813e-02,  9.500001e-03],
 [ 4.800007e-02 ,-1.088813e-02, -9.500001e-03],
 [ 4.800007e-02 ,-1.088813e-02,  9.500001e-03],
 [ 4.800007e-02 , 1.088813e-02, -9.500001e-03],
 [ 4.800007e-02 , 1.088813e-02,  9.500001e-03]])
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
    torch.manual_seed(0)
    #meta = np.load(os.path.join(cfg.demo_path, 'meta.npy'), allow_pickle=True).item()
    demo_images = glob.glob(cfg.demo_path + '/*jpg')
    network = make_network(cfg).cuda()
    print(cfg.model_dir)
    print(cfg.test.epoch)
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    visualizer = make_visualizer(cfg)

    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    
    for demo_image in demo_images:
        demo_image_ = np.array(Image.open(demo_image))    
        demo_image = np.array(Image.open(demo_image)).astype(np.float32)
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
            cv2.imshow('im_out', im_out) 
      
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
            print("corner_2d_pred",corner_2d_pred)
            print("pose_pred",pose_pred)
            ack ={'corner_2d_pred': corner_2d_pred.tolist(), 'pose_pred':pose_pred.tolist()}
            #cv2.namedWindow("seg", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            #cv2.resizeWindow("seg", demo_image_.shape[1], demo_image_.shape[0])
            cv2.imshow("RGB",demo_image_)
            cv2.waitKey(1)
            import time
            time.sleep(0.01)

# After the loop release the cap object 
    vid.release() 
#out.release()
# Destroy all the windows 
    cv2.destroyAllWindows() 
if __name__ == '__main__':
    globals()['run_'+args.type]()

