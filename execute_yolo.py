import shutil
import os
import random
from sklearn.model_selection import train_test_split

def create_folder(path):
    if os.path.exists(path) == False:
        os.mkdir(path)
    else:
        for files in os.listdir(path):
            path_remove = os.path.join(path, files)
            try:
                shutil.rmtree(path_remove)
            except OSError:
                os.remove(path_remove)
                
# moves files into destination folder
def move_files(files, destination):
    for f in files:
        try:
            shutil.copy(f, destination)
        except:
            print(f)
            assert False
            

def get_train_val_test(dataset_size = 10795):       
    path = os.getcwd()
    # parent_path = os.path.dirname(path)        
    imgs_root = os.path.join(path, "Synthetic_Dataset/YOLO/Images")
    labels_root = os.path.join(path, "Synthetic_Dataset/YOLO/Labels")
    
    paths_images = []
    paths_labels = []
    
    for img_file in os.listdir(imgs_root):
        img_path = os.path.join(imgs_root, img_file)
        label = img_file.replace('tif', 'txt')
        label_path = os.path.join(labels_root, label)
        paths_images.append(img_path)
        paths_labels.append(label_path)
    
    paths = list(zip(paths_images, paths_labels))
    random.shuffle(paths)
    paths_images, paths_labels = zip(*paths)
    
    
    paths_images = paths_images[:dataset_size]
    paths_labels = paths_labels[:dataset_size]
    
    train_images, val_images, train_annotations, val_annotations = train_test_split(paths_images, paths_labels, test_size = 0.1, random_state = 1)
    
    # val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0, random_state = 1)
    test_images, test_annotations = [], []
    
    path_yolo_dataset = os.path.join(path, 'yolov5_dataset_{}'.format(dataset_size)); create_folder(path_yolo_dataset)
    
    path_yolo_imgs = os.path.join(path_yolo_dataset, 'images'); create_folder(path_yolo_imgs)
    path_yolo_imgs_test = os.path.join(path_yolo_dataset, 'images/test'); create_folder(path_yolo_imgs_test)
    path_yolo_imgs_train = os.path.join(path_yolo_dataset, 'images/train'); create_folder(path_yolo_imgs_train)
    path_yolo_imgs_val = os.path.join(path_yolo_dataset, 'images/val'); create_folder(path_yolo_imgs_val)
    
    path_yolo_labels = os.path.join(path_yolo_dataset, 'labels'); create_folder(path_yolo_labels)
    path_yolo_labels_test = os.path.join(path_yolo_dataset, 'labels/test'); create_folder(path_yolo_labels_test)
    path_yolo_labels_train = os.path.join(path_yolo_dataset, 'labels/train'); create_folder(path_yolo_labels_train)
    path_yolo_labels_val = os.path.join(path_yolo_dataset, 'labels/val'); create_folder(path_yolo_labels_val)
    
    
    move_files(train_images, path_yolo_imgs_train)
    move_files(val_images, path_yolo_imgs_val)
    move_files(test_images, path_yolo_imgs_test)
    move_files(train_annotations, path_yolo_labels_train)
    move_files(val_annotations, path_yolo_labels_val)
    move_files(test_annotations, path_yolo_labels_test)

# python train.py --img 640 --cfg models/yolov5s.yaml --hyp data/hyps/hyp.scratch-low.yaml --batch 32 --epochs 100 --data data/custom_dataset.yaml --weights yolov5s.pt --workers 24 --name custom_log
# python detect.py --source "C:\Users\Utilizador\Desktop\Tese\Tarefas\Tarefa 5 - Resultados\Yolo_results\yolov5_dataset\images\test" --weights runs/train/custom_log13/weights/best.pt --conf 0.25 --name detection_log --save-crop --save-txt --save-conf --line-thickness 1 --hide-labels
# python val.py --weights runs/train/custom_log13/weights/best.pt --conf 0.25 --name detection_log --save-txt --save-conf --task test --data "data/custom_dataset.yaml" --iou-thres 0.6

    # parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    # parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    # parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    # parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    # parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    # parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    # parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--view-img', action='store_true', help='show results')
    # parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    # parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    # parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # parser.add_argument('--augment', action='store_true', help='augmented inference')
    # parser.add_argument('--visualize', action='store_true', help='visualize features')
    # parser.add_argument('--update', action='store_true', help='update all models')
    # parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    # parser.add_argument('--name', default='exp', help='save results to project/name')
    # parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    # parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    # parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    # parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    # parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')