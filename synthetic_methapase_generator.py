import os
import numpy as np
import cv2 as cv
import random
from random import randrange
import dataset_functions as cf # chromosome functions
import time
import json
import shutil

def erase_images(path, erase_images = True):
    path_images = os.path.join(path, r"Synthetic_Dataset/Synthetic_Images")
    
    if erase_images == True:
        foldernames_images = next(os.walk(path_images), (None, None, []))[1]
        for folder in foldernames_images:
            path_image_folder = os.path.join(path_images, folder)
            shutil.rmtree(path_image_folder)
            

                
def generate_dataset(path, synthetic_img, nucleolus_exists = True, nr_chromosomes = 46, \
                     nr_other_objects = 50, nr_other_objects_random = True, nr_other_objects_random_max = 50, \
                         width = 1024, height = 1376, iterations_ch2 = 10, kernel_ch2 = 2, \
                             iterations_ch1 = 10, kernel_ch1 = 2):
    ####################################################################################################
    ########################################### DIRECTORIES ###########################################
    ####################################################################################################
    # gets current path
    # path = os.getcwd()
    # gets cropped folder with all the individual objects
    path_cropped = os.path.join(path, r"Synthetic_Dataset/Cropped") 
    # chromosomes path
    path_chromosomes = os.path.join(path_cropped, "Chromosomes")
    # nucleolus path
    path_nucleolus = os.path.join(path_cropped, "Nucleolus") 
    # other objects path
    path_other_objects = os.path.join(path_cropped, "Other_Objects") 
    # gets folder in which the images will be saved
    path_imgs = os.path.join(path, r"Synthetic_Dataset/Synthetic_Images") 
    
    
    ####################################################################################################
    ########################################### INPUTS ###########################################
    ####################################################################################################
    # desired number of celular images
    # nr_synthetic_img = 1
    # desired nucleolus
    # nucleolus_exists = True
    # desired number of chromosomes per image (until 46 per image)
    # nr_chromosomes = 5
    # desired number of noisy objects per image (max 50 per image)
    # nr_other_objects = 50
    # nr_other_objects_random = True # if True, generates randomly a number between 0 and nr_other_objects_random_max
    # nr_other_objects_random_max = 50
    # desired resolution ( careful, because of pasting the crops...)
    # width = 1024
    # height = 1376
    resolution = [width, height]
    # blur over ch2
    # iterations_ch2 = 0; kernel_ch2 = 2
    # blur over ch1
    # iterations_ch1 = 0; kernel_ch1 = 2
    # number of existing synthetic images
    nr_synthetic_img_total = len(next(os.walk(path_imgs), (None, None, []))[1])
    # date
    date = cf.get_date()
    
    
    ####################################################################################################
    ########################################### SYNTETHIC IMAGE GENERATION ###########################################
    ####################################################################################################
    
    # for synthetic_img in range(nr_synthetic_img_total, nr_synthetic_img_total + nr_synthetic_img):
    print("\n\n------------------- Image {} generation --------------------\n\n ".format(synthetic_img))
    start_script_time = time.time()
    
    ########################################### DIRECTORIES ###########################################
    # creates directories for syntethic image
    path_img = os.path.join(path_imgs, str(synthetic_img)) 
    path_img_masks = os.path.join(path_img, r'Masks') 
    path_img_chromosomes = os.path.join(path_img, r'Chromosomes')
    path_img_nucleolus = os.path.join(path_img, r'Nucleolus') 
    path_img_other_objects = os.path.join(path_img, r'Other_Objects') 
    # meter segurança para a criacao de ficheiros
    os.mkdir(path_img);os.mkdir(path_img_masks);os.mkdir(path_img_chromosomes);os.mkdir(path_img_nucleolus);os.mkdir(path_img_other_objects)
    
    
    ########################################### INITIALIZE IMAGES ###########################################
    # saves syntetic celular image
    img = np.zeros((width,height), dtype='uint8')
    img[:,:] = 255
    
    # saves image mask
    img_mask = np.zeros((width,height), dtype='uint8')
    img_mask[:,:] = 255
    
    # saves overlap mask
    overlap_mask = np.zeros((width,height), dtype='uint8')
    overlap_mask[:,:] = 255
    
    # shows chromossomes and overlap mask in color
    img_colors = np.zeros((width,height,3), dtype='uint8')
    
    # shows chromossomes and its labels in blue
    img_labels = np.zeros((width,height,3), dtype='uint8')
    
    ########################################### INITIALIZE LABEL DICTIONARY ###########################################
    # it will save the retangular bbox of each chromosome and/or other objects
    nr_clusters = 0
    labels = {"date": date, "type": 'Methapase Syntetic Image Number {}'.format(str(synthetic_img)), "Nr_Chromosomes": nr_chromosomes, "Nr_Clusters": nr_clusters, "Nucleolus_exists": nucleolus_exists, "Nr_Random_Objects": nr_other_objects, "shapes": [], \
              "imagePath": 'Synthetic_Metaphase_{}.tif'.format(str(synthetic_img)), "imageHeight": width, "imageWidth": height}
    
                
    ########################################### LISTS WITH CHROMOSOMES THAT WILL BE PASTED ###########################################
    # saves which chromosomes are going to be pasted
    chromosomes_list = [n+1 for n in range(23)]; chromosomes_list.extend([n+1 for n in range(22)]) # because 2n cells
    # chooses man or woman
    gender = randrange(2)
    if gender == 1: # man
        chromosomes_list.append(24)
    else:
        chromosomes_list.append(23)
    # randomizes the order of pasted chromosomes, then it is chosen last element and eliminated when pasted
    random.shuffle(chromosomes_list)
    # checks if the homologous has been pasted (also saves YY)
    chromosomes_list_homologous = ['' for h in range(25)]  
    # saves the colour of each chromosome
    random_colour_list = []
    
    ########################################### PASTE NUCLEOLUS ###########################################
    start_paste_time = time.time()
    img, img_mask, nucleolus_position, nucleolus_dimensions, random_colour_list, shape_label = cf.paste_nucleolus(img, img_mask, nucleolus_exists, path_nucleolus, path_img_nucleolus, resolution, random_colour_list)
    # updates label
    if type(shape_label) == dict:
        labels["shapes"].append(shape_label)
        end_paste_time = time.time()
        print("Duration of pasting nucleolus: {} seconds \n\n ".format(end_paste_time-start_paste_time))
    
    ########################################### PASTE FIRST CHROMOSOME ###########################################
    if nr_chromosomes > 0:
        start_paste_time = time.time()
        chromosome_class = chromosomes_list[-1]
        img, img_mask, chromosome_filename, shape_label, random_colour_list = \
            cf.paste_first_chromosome(img, img_mask, chromosome_class, path_chromosomes, path_img_chromosomes, nucleolus_exists, nucleolus_position, nucleolus_dimensions, resolution, random_colour_list)
        # updates label
        labels["shapes"].append(shape_label)
        
        chromosomes_list_homologous[chromosome_class] = chromosome_filename # saves the chromosome filename which was pasted
        chromosomes_list.pop() # eliminates the chromosome number which was pasted
        # saves mask of first chromosome in the colours mask
        img_colors[:,:,:] = 0
        img_colors[:,:,0] = img_mask[:,:]
        img_colors[:,:,1] = img_mask[:,:]
        img_colors[:,:,2] = img_mask[:,:]
        
        end_paste_time = time.time()
        print("Duration of pasting chromosome 1: {} seconds \n\n ".format(end_paste_time-start_paste_time))
    
    ########################################### PASTE REMAINING CHROMOSOMES ###########################################
    if nr_chromosomes > 46:
            nr_chromosomes = 46 # limit of 46 chromosomes per image
    for chromosome_number in range(nr_chromosomes-1):
        start_paste_time = time.time()
        
        # saves overlap mask of pasted chromosome, ch2, over the rest of the image, ch1
        overlap_mask_ch2 = np.zeros((width,height), dtype='uint8')
        overlap_mask_ch2[:,:] = 255
        
        # saves overlap mask of the image, ch1, over the pasted chromosome, ch2
        overlap_mask_ch1 = np.zeros((width,height), dtype='uint8')
        overlap_mask_ch1[:,:] = 255
        
        # chromosome that is being pasted
        random.shuffle(chromosomes_list)
        chromosome_class = chromosomes_list[-1]
        path_chromosome2 = os.path.join(path_chromosomes, str(chromosome_class))
        
        # checks if the homologous has been pasted
        path_chromosome2, chromosome2_filename, chromosomes_list_homologous = cf.check_homologous(chromosomes_list_homologous, chromosome_class, path_chromosome2)
        
        # pastes second chromosome
        # pode se fazer a mesma coisa e eliminar o cromossoma que ja foi usado, assim nao aparece em imagens futuras.... (ideia)
        img, img_mask, overlap_mask, overlap_mask_ch1, overlap_mask_ch2, chromosome2, ch2_mask, dimensions, position, random2_color_mask, shape_label, random_colour_list, cluster = \
            cf.paste_remaining_chromosome(img, img_mask, overlap_mask, overlap_mask_ch1, overlap_mask_ch2, path_chromosome2, chromosome2_filename, path_img_chromosomes, resolution, random_colour_list, labels)
        # updates label
        if cluster == True: nr_clusters+=1
        labels["shapes"].append(shape_label)
        chromosomes_list.pop() # eliminates the chromosome number which was pasted
        
        # erosion of chromosome mask
        kernel = np.ones((5,5),np.uint8)
        erosion = cv.erode(ch2_mask,kernel,iterations = 3)
        
        # makes a gradient of overlap mask of the pasted chromosome, saving only the frontier pixels
        gradient = cv.morphologyEx(overlap_mask_ch2, cv.MORPH_GRADIENT, np.ones((9,9),np.uint8))
        
    
        # blurs the frontier pixels ch2 over ch1
        img = cf.blur_frontier2(chromosome2, gradient, img, erosion, position, dimensions, iterations_ch2, kernel_ch2)

        
        # blurs the frontier pixels ch1 over ch2
        img = cf.blur_frontier1(overlap_mask_ch1, img, iterations_ch1, kernel_ch1)

        # updates colours mask    
        img_colors = cf.make_colour_mask(img, img_colors, gradient, chromosome2, overlap_mask_ch1, erosion, position, dimensions, ch2_mask, random2_color_mask)
        
        end_paste_time = time.time()
        
        print("Duration of pasting chromosome number {}: {} seconds \n\n ".format(chromosome_number+2, end_paste_time-start_paste_time))
    
    
    ########################################### PASTE OTHER OBJECTS ###########################################
    # checks number of other objects
    if nr_other_objects_random == True:
        if nr_other_objects_random_max > 50:
            nr_other_objects_random_max = 50 # limit of 50 other objects
        nr_other_objects = randrange(nr_other_objects_random_max) # random number of other objects
    elif nr_other_objects_random == False:
        if nr_other_objects > 50:
            nr_other_objects = 50 # limit of 50 other objects
    # pastes other objects        
    objs_not_found = 0
    for nr_obj in range(nr_other_objects):
        start_paste_time = time.time()
        img, img_mask, random_colour_list, shape_label = cf.paste_other_objects(img, img_mask, path_other_objects, path_img_other_objects, resolution, random_colour_list)
        if type(shape_label) == int:
            end_paste_time = time.time()
            print("Noisy object number {} wasn't pasted because random location was not found".format(nr_obj))
            objs_not_found = objs_not_found + 1
        else:
            # updates label
            labels["shapes"].append(shape_label)
            end_paste_time = time.time()
            print("Duration of pasting noisy object number {}: {} seconds \n\n ".format(nr_obj, end_paste_time-start_paste_time))
    print("Number of pasted noisy objects: {} \n".format(nr_other_objects-objs_not_found))
    ########################################### VALIDATE/WRITE LABELS AND SAVE IMAGES ###########################################    
    # updates labels
    labels["Nr_Chromosomes"] = nr_chromosomes
    labels["Nr_Clusters"] = nr_clusters
    labels["Nr_Random_Objects"] = nr_other_objects
    # validates labels
    img_labels[:,:,0] = img[:,:]
    img_labels[:,:,1] = img[:,:]
    img_labels[:,:,2] = img[:,:]
    img_labels = cf.validate_labels(img_labels, labels)
    # saves syntethic image and its masks
    cv.imwrite(os.path.join(path_img, 'Synthetic_Metaphase_{}.tif'.format(str(synthetic_img))), img)   
    cv.imwrite(os.path.join(path_img_masks, 'Synthetic_Metaphase_{}_Mask.tif'.format(str(synthetic_img))), img_mask)  
    cv.imwrite(os.path.join(path_img_masks, 'Synthetic_Metaphase_{}_Colour_Mask.tif'.format(str(synthetic_img))), img_colors)  
    cv.imwrite(os.path.join(path_img_masks, 'Synthetic_Metaphase_{}_Overlap.tif'.format(str(synthetic_img))), overlap_mask)    
    cv.imwrite(os.path.join(path_img_masks, 'Synthetic_Metaphase_{}_Labels.tif'.format(str(synthetic_img))), img_labels)               
    # writes and saves .json file
    json_object = json.dumps(labels, indent = 4)
    path_json_file = os.path.join(path_img, 'Label_Synthetic_Metaphase_{}.json'.format(str(synthetic_img)))
    with open(path_json_file, "w") as outfile:
        outfile.write(json_object)
    
    
    end_script_time = time.time()
    print("\n\nImage {} generation completed: {} seconds \n\n---------------------------------------\n\n ".format(synthetic_img, end_script_time-start_script_time))