import dataset_functions as cf # chromosome functions
import os
import cv2 as cv

def erase_cropped(path, erase_chro = True, erase_nucl = True, erase_obj = True):
    path_chromosomes = os.path.join(path, r"Synthetic_Dataset/Cropped/Chromosomes") 
    path_nucleolus = os.path.join(path, r"Synthetic_Dataset/Cropped/Nucleolus") 
    path_other_objects = os.path.join(path, r"Synthetic_Dataset/Cropped/Other_Objects") 
    if erase_chro == True:
        foldernames_chromosomes = next(os.walk(path_chromosomes), (None, None, []))[1]
        for folder in foldernames_chromosomes:
            path_chromosome = os.path.join(path_chromosomes, folder)
            filenames_chromosomes = next(os.walk(path_chromosome), (None, None, []))[2]
            for filename in filenames_chromosomes:
                filepath_chromosome = os.path.join(path_chromosome, filename)
                os.remove(filepath_chromosome)
                
    if erase_nucl == True:
        filenames_nucleolus = next(os.walk(path_nucleolus), (None, None, []))[2]
        for filename in filenames_nucleolus:
                filepath_nucleolus = os.path.join(path_nucleolus, filename)
                os.remove(filepath_nucleolus)
        
    if erase_obj == True:
        filenames_other_objects = next(os.walk(path_other_objects), (None, None, []))[2]
        for filename in filenames_other_objects:
                filepath_other_objects = os.path.join(path_other_objects, filename)
                os.remove(filepath_other_objects)    

def extract_cropped_images(path, extract_chr = True, extract_nucl = True, extract_obj = True):
    print("\n\nExtracting cropped chromosomes...")
    # gets labelme folder with thedatasets from each labeled image
    path_labelme = os.path.join(path, r"Synthetic_Dataset/LabelMe_Images") 
    # gets all the datasets generated in label me
    datasets_labelme = next(os.walk(path_labelme), (None, None, []))[1] # 1 gets folders
    # gets cropped folder in which objects will be saved
    path_chromosomes = os.path.join(path, r"Synthetic_Dataset/Cropped/Chromosomes") 
    path_nucleolus = os.path.join(path, r"Synthetic_Dataset/Cropped/Nucleolus") 
    path_other_objects = os.path.join(path, r"Synthetic_Dataset/Cropped/Other_Objects") 
    

    for data in datasets_labelme:
        print("      ...")
        # gets each dataset path
        data_path = os.path.join(path_labelme, data)
        # gets files in each dataset: [.json, .tif, img.png, label.png, label_names.txt, 'label_viz.png]
        data_filenames = next(os.walk(data_path), (None, None, []))[2] # 2 gets files
        for file in data_filenames:
            # gets json filepath
            if '.json' in file:    
                json_filepath = os.path.join(data_path, file) # .json
            # gets tif filepath
            elif '.tif' in file:    
                image_path = os.path.join(data_path, file) # .tif

        # gets data from json
        data = cf.open_json(json_filepath)
        
        # reads .tif image and gets its pixel' values
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        
        print(json_filepath)
        # translates data in json to individual chromosomes
        for shape in data['shapes']:
            if (extract_chr == True and shape['group_id'] == 1): # chromosome
                # gets chromosome    
                chromosome = cf.crop_obj(img, shape)
                # checks which is the chromosome number and gives its directory
                path_chromosomes_number = cf.define_path_chromosome(shape, path_chromosomes)
                print("{} to {}\n".format(shape['label'],path_chromosomes_number))
                # rotates each chromosome from 0 to 345 degrees, with a 15 degree step
                for angle in range(0,360,15):
                    rotated_chromosome = cf.rotate_bound(chromosome, angle)
                    # saves each rotated chromosome
                    rotated_chromosome_path = os.path.join(path_chromosomes_number, "{}_rotated_{}.tif".format(shape['label'], angle)) 
                    cv.imwrite(rotated_chromosome_path, rotated_chromosome)
            elif (extract_nucl == True and shape['group_id'] == 2): # nucleolus
                # crops individual nucleolus from image
                nucleolus = cf.crop_obj(img, shape)
                print("{}\n".format(shape['label']))
                # saves each nucleolus
                nucleolus_path = os.path.join(path_nucleolus, "{}.tif".format(shape['label'])) 
                cv.imwrite(nucleolus_path, nucleolus)
            elif (extract_obj == True and shape['group_id']) == 3: # other objects
                # crops individual object from image
                other_object = cf.crop_obj(img, shape)
                print("{}\n".format(shape['label']))
                # rotates each object from 0 to 345 degrees, with a 15 degree step
                for angle in range(0,360,15):
                    rotated_other_object = cf.rotate_bound(other_object, angle)
                    # saves each rotated object 
                    rotated_other_object_path = os.path.join(path_other_objects, "{}_rotated_{}.tif".format(shape['label'], angle)) 
                    cv.imwrite(rotated_other_object_path, rotated_other_object)
                    
    print("\n\nCropped dataset created successfuly!\n\n")