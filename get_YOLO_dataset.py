import os
import cv2 as cv
import dataset_functions as cf # chromosome functions

def erase_yolo(path, erase_yolo = True):
    path_yolo = os.path.join(path, r"Synthetic_Dataset/YOLO")
    
    if erase_yolo == True:
        foldernames_yolo = next(os.walk(path_yolo), (None, None, []))[1]
        for folder in foldernames_yolo:
            path_yolo_folder = os.path.join(path_yolo, folder)
            filenames = next(os.walk(path_yolo_folder), (None, None, []))[2]
            for filename in filenames:
                filepath = os.path.join(path_yolo_folder, filename)
                os.remove(filepath)
            
def convert_labels_to_YOLO(path, label_chromosomes = True, label_nucleolus = False, label_other_objects = False):
    print("\n\nCreating YOLO dataset...")
    ####################################################################################################
    ########################################### DIRECTORIES ###########################################
    ####################################################################################################
    # # gets current path
    # path = os.getcwd()
    # gets folder with all the syntethic images created
    path_dataset = os.path.join(path, r"Synthetic_Dataset/Synthetic_Images") 
    foldernames_images = next(os.walk(path_dataset), (None, None, []))[1]
    # directory in which labels will be saved
    path_YOLO_labels = os.path.join(path, r"Synthetic_Dataset/YOLO/Labels") 
    # directory in which images will be saved
    path_YOLO_images = os.path.join(path, r"Synthetic_Dataset/YOLO/Images") 
    # directory in which validation images will be saved
    path_YOLO_validations = os.path.join(path, r"Synthetic_Dataset/YOLO/Validations") 

    
    for img_folder in foldernames_images:
        print("      ... {}".format(img_folder))
        img_folder_path = os.path.join(path_dataset, img_folder) 
        foldernames_images = next(os.walk(img_folder_path), (None, None, []))[2]
        # label path
        label_path = os.path.join(img_folder_path, foldernames_images[0]) 
        # image path
        img_path = os.path.join(img_folder_path, foldernames_images[1])
        # syntethic image name
        img_name = foldernames_images[0][6:-5]
        
        # gets data from json
        labels = cf.open_json(label_path)
        
        # gets yolo .txt
        yolo_labels = cf.convert_label_to_yolo(labels)
    
        # gets image
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        
        # saves label
        label_filename = img_name + '.txt'
        path_YOLO_label = os.path.join(path_YOLO_labels, label_filename) 
        f = open(path_YOLO_label, "w")
        for label in yolo_labels:
            if (label_chromosomes == True and label[0] == 1-1): # only saves chromosomes labels
                for item in label:
                    f.write(str(item) + " ")
                f.write("\n")
            elif (label_nucleolus == True and label[0] == 2-1): # only saves nucleolus labels
                for item in label:
                    f.write(str(item) + " ")
                f.write("\n")
            elif (label_other_objects == True and label[0] == 3-1): # only saves other_objects labels
                for item in label:
                    f.write(str(item) + " ")
                f.write("\n")
        f.close()
    
        # saves image
        img_filename = img_name + '.tif'
        path_YOLO_image = os.path.join(path_YOLO_images, img_filename) 
        cv.imwrite(path_YOLO_image, img)
        
        # validates yolo labels
        img = cf.validate_YOLO_labels(img, yolo_labels, label_chromosomes, label_nucleolus, label_other_objects)
        path_YOLO_validation = os.path.join(path_YOLO_validations, img_filename) 
        cv.imwrite(path_YOLO_validation, img)
        
        
    print("\n\nYOLO dataset created successfuly!\n\n")