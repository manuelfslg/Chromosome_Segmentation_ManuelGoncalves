import json
import os
from random import randrange
import random
import cv2 as cv
import numpy as np
from datetime import date
import shutil
import matplotlib.pyplot as plt

# opens json file
def open_json(filepath):
    # Opening JSON file
    f = open(filepath, encoding='utf-8') #
    # returns JSON object as a dictionary
    data = json.load(f) #
    return data

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

# crops object from the image in regards to its bbox from labelme
def crop_obj(img, shape):
    # gets rectangle bbox vertices
    x_min = int(shape['points'][0][0])
    y_min = int(shape['points'][0][1])
    x_max = int(shape['points'][1][0])
    y_max = int(shape['points'][1][1])
    # crops individual object from image
    obj = img[y_min:y_max, x_min:x_max]
    return obj


def define_path_chromosome(shape, path_chromosomes):
    if shape['label'][:4] == 'ch01': folder = 1
    if shape['label'][:4] == 'ch02': folder = 2
    if shape['label'][:4] == 'ch03': folder = 3
    if shape['label'][:4] == 'ch04': folder = 4
    if shape['label'][:4] == 'ch05': folder = 5
    if shape['label'][:4] == 'ch06': folder = 6
    if shape['label'][:4] == 'ch07': folder = 7
    if shape['label'][:4] == 'ch08': folder = 8
    if shape['label'][:4] == 'ch09': folder = 9
    if shape['label'][:4] == 'ch10': folder = 10
    if shape['label'][:4] == 'ch11': folder = 11
    if shape['label'][:4] == 'ch12': folder = 12
    if shape['label'][:4] == 'ch13': folder = 13
    if shape['label'][:4] == 'ch14': folder = 14
    if shape['label'][:4] == 'ch15': folder = 15
    if shape['label'][:4] == 'ch16': folder = 16
    if shape['label'][:4] == 'ch17': folder = 17
    if shape['label'][:4] == 'ch18': folder = 18
    if shape['label'][:4] == 'ch19': folder = 19
    if shape['label'][:4] == 'ch20': folder = 20
    if shape['label'][:4] == 'ch21': folder = 21
    if shape['label'][:4] == 'ch22': folder = 22
    if shape['label'][:4] == 'chXX': folder = 23
    if shape['label'][:4] == 'chYY': folder = 24
    chromosome_path = os.path.join(path_chromosomes, str(folder))
    return chromosome_path

def my_blur(img, pixel, kernel): # meter validacao na existencia de pixeis
    kernel = kernel - 1
    i_begin = pixel[0] - kernel
    j_begin = pixel[1] - kernel
    soma = 0
    pixels = (kernel*2 + 1) ** 2
    
    for i in range(2*kernel+1):
        for j in range(2*kernel+1):
            if i_begin+i < img.shape[0] and j_begin+j < img.shape[1]:
                soma+= int(img[i_begin+i,j_begin+j])
            else:
                pixels = pixels - 1

    new_pixel_value = int(soma/pixels)
    return new_pixel_value

def homologous_name(my_str, letter):
    # substitutes for homologous
    my_str_split = my_str.split('_')
    my_str_new = my_str_split[0] + '_' + letter + "_" +  my_str_split[2] + "_" + my_str_split[3] + "_"
    my_list = list(my_str_new)
    # randomly chooses one of the rotated angles of the homologous
    angles = [str(angle) for angle in range(0,360,15)]
    random = randrange(24)
    angle = angles[random]
    angle_list = list(angle)
    # joins information
    my_list.extend(angle_list)
    new_str = ''.join(my_list) + '.tif'
    return new_str

def define_nucleolus_location(nucleolus, nucleolus_filename, resolution):
    # should have a check point to see if the image size is bigger than the nucleolus size
    location = nucleolus_filename.split("_")[2]
    if location == 'ltc': # left top corner
        nucleolus_position_x = 0
        nucleolus_position_y = 0  
    elif location == 'rtc':   # right top corner 
        nucleolus_position_x = 0
        nucleolus_position_y = resolution[1]-nucleolus.shape[1]
    elif location == 'ldc':    # left down corner
        nucleolus_position_x = resolution[0]-nucleolus.shape[0]
        nucleolus_position_y = 0
    elif location == 'rdc':    # right down corner
        nucleolus_position_x = resolution[0]-nucleolus.shape[0]
        nucleolus_position_y = resolution[1]-nucleolus.shape[1]
    elif location == 'l':    # left
        nucleolus_position_x = randrange(resolution[0]-nucleolus.shape[0])
        nucleolus_position_y = 0
    elif location == 'r':    # right
        nucleolus_position_x = randrange(resolution[0]-nucleolus.shape[0])
        nucleolus_position_y = resolution[1]-nucleolus.shape[1]
    elif location == 't':    # top
        nucleolus_position_x = 0
        nucleolus_position_y = randrange(resolution[1]-nucleolus.shape[1])
    elif location == 'd':    # down
        nucleolus_position_x = resolution[0]-nucleolus.shape[0]
        nucleolus_position_y = randrange(resolution[1]-nucleolus.shape[1])
    elif location == 'm':    # middle
        nucleolus_position_x = randrange(resolution[0]-nucleolus.shape[0])
        nucleolus_position_y = randrange(resolution[1]-nucleolus.shape[1])
    positions = [nucleolus_position_x, nucleolus_position_y]
    return positions


def paste_nucleolus(img, img_mask, nucleolus_exists, path_nucleolus, path_img_nucleolus, resolution, random_colour_list):
    if nucleolus_exists == True:
        # paste nucleolus
        filenames_nucleolus = next(os.walk(path_nucleolus), (None, None, []))[2]
        random.shuffle(filenames_nucleolus)
        nucleolus_filename = filenames_nucleolus[-1]
        path_nucleolus1 = os.path.join(path_nucleolus, nucleolus_filename)
        # pode se fazer a mesma coisa e eliminar o cromossoma que ja foi usado, assim nao aparece em imagens futuras.... (ideia)
        nucleolus = cv.imread(path_nucleolus1 , cv.IMREAD_GRAYSCALE)
        cv.imwrite(os.path.join(path_img_nucleolus, '{}.tif'.format(nucleolus_filename)), nucleolus)
        print("Pasting nucleolus {}\n".format(nucleolus_filename))
        
        # generate new random colour for nucleolus mask
        random_colour_list, random_color_mask = check_random_color(random_colour_list)
    
        paste_width_nucleolus = nucleolus.shape[0]
        paste_height_nucleolus = nucleolus.shape[1]
        nucleolus_dimensions = [paste_width_nucleolus, paste_height_nucleolus]
        
        nucleolus_position = define_nucleolus_location(nucleolus, nucleolus_filename, resolution)
        nucleolus_position_x = nucleolus_position[0]
        nucleolus_position_y = nucleolus_position[1]
        # fisrt paste
        for i in range(nucleolus_position_x, nucleolus_position_x + paste_width_nucleolus):
            for j in range(nucleolus_position_y, nucleolus_position_y + paste_height_nucleolus):
                if nucleolus[i-nucleolus_position_x,j-nucleolus_position_y] != 255: # only copies pixels from chromosome, and not the white background
                    img[i,j] = nucleolus[i-nucleolus_position_x,j-nucleolus_position_y]
                    img_mask[i,j] = random_color_mask
                    
        shape_label = create_label(nucleolus_filename, nucleolus_position_x, nucleolus_position_y, paste_width_nucleolus, paste_height_nucleolus, random_color_mask, 2)
    else:
        shape_label = -1
        nucleolus_position = -1
        nucleolus_dimensions = -1
    return img, img_mask, nucleolus_position, nucleolus_dimensions, random_colour_list, shape_label


def paste_first_chromosome(img, img_mask, chromosome_class, path_chromosomes, path_img_chromosomes, nucleolus_exists, nucleolus_position, nucleolus_dimensions, resolution, random_colour_list):
    path_chromosome1 = os.path.join(path_chromosomes, str(chromosome_class))
    filenames_chromosomes = next(os.walk(path_chromosome1), (None, None, []))[2]
    random.shuffle(filenames_chromosomes)
    chromosome_filename = filenames_chromosomes[-1]
    path_chromosome1 = os.path.join(path_chromosome1, chromosome_filename)
    # pode se fazer a mesma coisa e eliminar o cromossoma que ja foi usado, assim nao aparece em imagens futuras.... (ideia)
    chromosome1 = cv.imread(path_chromosome1, cv.IMREAD_GRAYSCALE)
    cv.imwrite(os.path.join(path_img_chromosomes, '{}.tif'.format(chromosome_filename)), chromosome1)
    print("Pasting chromosome {}\n".format(chromosome_filename))
    
    # generate new random colour for ch1 mask
    random_colour_list, random1_color_mask = check_random_color(random_colour_list)
    
    paste_width_1 = chromosome1.shape[0]
    paste_height_1 = chromosome1.shape[1]
    
    # checks if nucleolus is under chromosome. It cannot be in the first paste, because the algorithm doesn't smooth the edges
    if nucleolus_exists == True:
        nucleolus = True
        while nucleolus == True:
            random1_position_x = randrange(resolution[0]-paste_width_1)
            random1_position_y = randrange(resolution[1]-paste_height_1)
            # xb1 > xa2 # xb2 < xa1 # yb1 > ya2 # yb2 < ya1
            if (random1_position_x > nucleolus_position[0] + nucleolus_dimensions[0] \
                or random1_position_x + paste_width_1 < nucleolus_position[0] \
                    or random1_position_y > nucleolus_position[1] + nucleolus_dimensions[1] \
                        or random1_position_y + paste_height_1 < nucleolus_position[1]): 
                nucleolus = False
    else:
        random1_position_x = randrange(resolution[0]-paste_width_1)
        random1_position_y = randrange(resolution[1]-paste_height_1)
        
    # fisrt paste
    for i in range(random1_position_x, random1_position_x + paste_width_1):
        for j in range(random1_position_y, random1_position_y + paste_height_1):
            if chromosome1[i-random1_position_x,j-random1_position_y] != 255: # only copies pixels from chromosome, and not the white background
                img[i,j] = chromosome1[i-random1_position_x,j-random1_position_y]
                img_mask[i,j] = random1_color_mask
    # label
    shape_label = create_label(chromosome_filename, random1_position_x, random1_position_y, paste_width_1, paste_height_1, random1_color_mask, 1)
    
    return img, img_mask, chromosome_filename, shape_label, random_colour_list

def check_homologous(chromosomes_list_homologous, chromosome_class, path_chromosome2):
    if chromosomes_list_homologous[chromosome_class] == '':   
            # random chromosome from the desired chromosome class
            filenames_chromosomes = next(os.walk(path_chromosome2), (None, None, []))[2]
            random.shuffle(filenames_chromosomes)
            chromosome2_filename = filenames_chromosomes[-1]
            path_chromosome2 = os.path.join(path_chromosome2, chromosome2_filename)
            chromosomes_list_homologous[chromosome_class] = chromosome2_filename
    else:
        # homologous chromosome
        homologous_filename = chromosomes_list_homologous[chromosome_class]
        homologous_filename_split = homologous_filename.split('_')
        
        if homologous_filename_split[1] == 'l': 
            chromosome2_filename = homologous_name(homologous_filename, 'r')
        else: 
            chromosome2_filename = homologous_name(homologous_filename, 'l')
        path_chromosome2 = os.path.join(path_chromosome2, chromosome2_filename)
    
    return path_chromosome2, chromosome2_filename, chromosomes_list_homologous

def paste_remaining_chromosome(img, img_mask, overlap_mask, overlap_mask_ch1, overlap_mask_ch2, path_chromosome2, chromosome2_filename, path_img_chromosomes, resolution, random_colour_list, labels):
    chromosome2 = cv.imread(path_chromosome2, cv.IMREAD_GRAYSCALE)
    print(path_chromosome2)
    cv.imwrite(os.path.join(path_img_chromosomes, '{}.tif'.format(chromosome2_filename)), chromosome2)
    print("Pasting chromosome {}\n".format(chromosome2_filename))
    
    # generate new random colour for ch2 mask
    random_colour_list, random2_color_mask = check_random_color(random_colour_list)
    
    # width and height of chromosomes
    paste_width_2 = chromosome2.shape[0]
    paste_height_2 = chromosome2.shape[1]
    # position of pasted chromosome
    random2_position_x = randrange(resolution[0]-paste_width_2)
    random2_position_y = randrange(resolution[1]-paste_height_2)

    # saves ch2 mask
    ch2_mask = np.zeros((random2_position_x + paste_width_2, random2_position_y + paste_height_2), dtype='uint8')
    ch2_mask[:,:] = 0
    # saves if there is overlap AKA cluster
    if labels['shapes'][0]["label"][:9] == "nucleolus":
        check_nucl_color = labels['shapes'][0]["mask_colour"]
    else:
        check_nucl_color = -1
    cluster = False
    
    for i in range(random2_position_x, random2_position_x + paste_width_2):
        for j in range(random2_position_y, random2_position_y + paste_height_2):
            if chromosome2[i-random2_position_x,j-random2_position_y] != 255: # only copies pixels from chromosome, and not the white background
                ch2_mask[i,j] = 255
                if img[i,j] != 255:
                    overlap_mask[i,j] = 0
                    overlap_mask_ch2[i,j] = 0
                    if chromosome2[i-random2_position_x,j-random2_position_y] <= 200:
                        img[i,j] = int(chromosome2[i-random2_position_x,j-random2_position_y]) 
                        img_mask[i,j] = random2_color_mask
                    elif chromosome2[i-random2_position_x,j-random2_position_y] > 200:
                        overlap_mask_ch1[i,j] = 0
                    # check if there is a cluster
                    if type(check_nucl_color) == int:
                        cluster = True
                    elif img_mask[i,j] != check_nucl_color:
                        cluster = True
                if img[i,j] == 255:
                    img[i,j] = chromosome2[i-random2_position_x,j-random2_position_y]
                    img_mask[i,j] = random2_color_mask
    #label
    shape_label = create_label(chromosome2_filename, random2_position_x, random2_position_y, paste_width_2, paste_height_2, random2_color_mask, 1)
    
    dimensions = [paste_width_2, paste_height_2]
    position = [random2_position_x, random2_position_y]
    
    return img, img_mask, overlap_mask, overlap_mask_ch1, overlap_mask_ch2, chromosome2, ch2_mask, dimensions, position, random2_color_mask, shape_label, random_colour_list, cluster

def paste_other_objects(img, img_mask, path_other_objects, path_img_other_objects, resolution, random_colour_list):
    # paste other object
    filenames_other_objects = next(os.walk(path_other_objects), (None, None, []))[2]
    random.shuffle(filenames_other_objects)
    other_object_filename = filenames_other_objects[-1]
    path_other_object = os.path.join(path_other_objects, other_object_filename)
    # pode se fazer a mesma coisa e eliminar o cromossoma que ja foi usado, assim nao aparece em imagens futuras.... (ideia)
    other_object = cv.imread(path_other_object, cv.IMREAD_GRAYSCALE)
    cv.imwrite(os.path.join(path_img_other_objects, '{}.tif'.format(other_object_filename)), other_object)
    print("Pasting noisy object {}\n".format(other_object_filename))
    
    # generate new random colour for other object mask
    random_colour_list, random_color_mask = check_random_color(random_colour_list, True)

    paste_width_other_object = other_object.shape[0]
    paste_height_other_object = other_object.shape[1]
    other_object_dimensions = [paste_width_other_object, paste_height_other_object]
    
    other_object_position_x, other_object_position_y = define_other_object_location(other_object, other_object_dimensions, resolution, img)
    
    if (other_object_position_x == -1 and other_object_position_y == -1):
        shape_label = -1
    else:
        # paste other object
        for i in range(other_object_position_x, other_object_position_x + paste_width_other_object):
            for j in range(other_object_position_y, other_object_position_y + paste_height_other_object):
                if other_object[i-other_object_position_x,j-other_object_position_y] != 255: # only copies pixels from chromosome, and not the white background
                    img[i,j] = other_object[i-other_object_position_x,j-other_object_position_y]
                    img_mask[i,j] = random_color_mask
                    
        shape_label = create_label(other_object_filename, other_object_position_x, other_object_position_y, paste_width_other_object, paste_height_other_object, random_color_mask, 3)

    return img, img_mask, random_colour_list, shape_label

def make_colour_mask(img, img_colors, gradient, chromosome2, overlap_mask_ch1, erosion, position, dimensions, ch2_mask, random2_color_mask):
    for i in range(0, position[0] + dimensions[0]):
        for j in range(0, position[1] + dimensions[1]):
            if ch2_mask[i,j] != 0:
                img_colors[i,j,0] = random2_color_mask
                img_colors[i,j,1] = random2_color_mask
                img_colors[i,j,2] = random2_color_mask
            if (gradient[i,j] == 255 and chromosome2[i-position[0],j-position[1]] <= 200 and img[i,j] <= 200):
                img_colors[i,j,0] = 0
                img_colors[i,j,1] = 255
                img_colors[i,j,2] = 0
            elif (overlap_mask_ch1[i,j] == 0):
                img_colors[i,j,0] = 0
                img_colors[i,j,1] = 0
                img_colors[i,j,2] = 255
            if erosion[i,j] == 255:
                img_colors[i,j,0] = 150
                img_colors[i,j,1] = 150
                img_colors[i,j,2] = 150
    return img_colors

def blur_frontier2(chromosome2, gradient, img, erosion, position, dimensions, iterations, kernel):
    for it in range(iterations):
        for i in range(0, position[0] + dimensions[0]):
            for j in range(0, position[1] + dimensions[1]):
                if (gradient[i,j] == 255 and chromosome2[i-position[0],j-position[1]] <= 200 and img[i,j] <= 200 and erosion[i,j] == 0):
                    img[i,j] = my_blur(img, [i,j], kernel)
    return img

def blur_frontier1(overlap_mask_ch1, img, iterations, kernel):
    for it in range(iterations):
        for i in range(0,overlap_mask_ch1.shape[0]):
            for j in range(0,overlap_mask_ch1.shape[1]):
                if (overlap_mask_ch1[i,j] == 0):
                    img[i,j] = my_blur(img, [i,j], kernel)
    return img


def get_date():
    today = date.today()
    today_string = '{}/{}/{}'.format(today.day, today.month, today.year)
    return today_string

def create_label(chromosome_filename, random_position_x, random_position_y, paste_width, paste_height, random_color_mask, group_id, bbox_margin = 3):
    bbox_initial = [random_position_y + bbox_margin, random_position_x + bbox_margin]
    bbox_ending = [random_position_y + paste_height - bbox_margin, random_position_x + paste_width - bbox_margin] 
    shape_label = {'label': chromosome_filename, 'points': [bbox_initial, bbox_ending], 'group_id': group_id, "shape_type": "rectangle", 'mask_colour': random_color_mask}
    return shape_label


def validate_labels(img_labels, labels):
    for shape in labels['shapes']:
        bbox_initial = shape['points'][0]
        bbox_ending = shape['points'][1]
        img_labels = cv.rectangle(img_labels, (bbox_initial[0], bbox_initial[1]), (bbox_ending[0], bbox_ending[1]), (255, 0, 0), 1)
    
    return img_labels

def check_random_color(random_colour_list, other_object = False):
    if other_object == False:
        colour_min = 25
        colour_max = 180
    elif other_object == True:
        colour_min = 181
        colour_max = 240
        
    random_colour_exists = True
    while random_colour_exists == True:
        random_colour_exists = False
        random_color_mask = randrange(colour_min, colour_max) # colour of random chromosome
        if random_color_mask in random_colour_list: # if there is other chromosome with the same colour
            random_colour_exists = True
    random_colour_list.append(random_color_mask)
    return random_colour_list, random_color_mask

def define_other_object_location(other_object, other_object_dimensions, resolution, img):
    chromosome = True
    random_loc = 0 # otherwise the program can get stuck pasting a random object
    random_loc_max = 50
    while (chromosome == True and random_loc < random_loc_max):
        random_loc = random_loc + 1
        random_position_x = randrange(resolution[0]-other_object_dimensions[0])
        random_position_y = randrange(resolution[1]-other_object_dimensions[1])
        # xb1 > xa2 # xb2 < xa1 # yb1 > ya2 # yb2 < ya1
        other_object_location = img[random_position_x:random_position_x+other_object_dimensions[0],random_position_y:random_position_y+other_object_dimensions[1]]
        if np.all(other_object_location == 255): 
            chromosome = False
        elif random_loc == random_loc_max:
            random_position_x = -1
            random_position_y = -1
    return random_position_x, random_position_y


def convert_label_to_yolo(labels):
    yolo_labels = []
    for shape in labels['shapes']:
        # label initial and ending bbox points
        bbox_initial = shape['points'][0]
        x_i = bbox_initial[0]
        y_i = bbox_initial[1]
        bbox_ending = shape['points'][1]
        x_f = bbox_ending[0]
        y_f = bbox_ending[1]
        
        # normalize width and height of bbox
        bbox_width = x_f - x_i
        bbox_width_normalized = bbox_width / labels['imageWidth']
        bbox_height = y_f - y_i
        bbox_height_normalized = bbox_height / labels['imageHeight']
        
        # get center and normalize it
        bbox_center_x = x_i + (x_f - x_i)/2
        bbox_center_x_normalized = bbox_center_x / labels['imageWidth']
        bbox_center_y = y_i + (y_f - y_i)/2
        bbox_center_y_normalized = bbox_center_y / labels['imageHeight']
        
        # yolo label
        yolo_label = [int(shape['group_id']-1),  bbox_center_x_normalized, bbox_center_y_normalized, bbox_width_normalized, bbox_height_normalized]    
        yolo_labels.append(yolo_label)
    return yolo_labels


def validate_YOLO_labels(img, labels, label_chromosomes, label_nucleolus, label_other_objects):
    # shows chromossomes and its labels in green
    img_width = img.shape[1]
    img_height = img.shape[0]
    img_labels = np.zeros((img_height, img_width, 3), dtype='uint8')
    img_labels[:,:,0] = img[:,:]
    img_labels[:,:,1] = img[:,:]
    img_labels[:,:,2] = img[:,:]
    
    for label in labels:
        if (label_chromosomes == True and label[0] == 1-1):
            x_center = label[1] * img_width
            y_center = label[2] * img_height
            label_width = label[3] * img_width
            label_height = label[4] * img_height
            x_i = int(x_center - label_width/2)
            x_f = int(x_center + label_width/2)
            y_i = int(y_center - label_height/2)
            y_f = int(y_center + label_height/2)
            bbox_x = [x_i, x_f]
            bbox_y = [y_i, y_f]
            img_labels = cv.rectangle(img_labels, (bbox_x[0], bbox_y[0]), (bbox_x[1], bbox_y[1]), (0, 255, 0), 1)
        elif (label_nucleolus == True and label[0] == 2-1):
            x_center = label[1] * img_width
            y_center = label[2] * img_height
            label_width = label[3] * img_width
            label_height = label[4] * img_height
            x_i = int(x_center - label_width/2)
            x_f = int(x_center + label_width/2)
            y_i = int(y_center - label_height/2)
            y_f = int(y_center + label_height/2)
            bbox_x = [x_i, x_f]
            bbox_y = [y_i, y_f]
            img_labels = cv.rectangle(img_labels, (bbox_x[0], bbox_y[0]), (bbox_x[1], bbox_y[1]), (0, 255, 0), 1)
        elif (label_other_objects == True and label[0] == 3-1):
            x_center = label[1] * img_width
            y_center = label[2] * img_height
            label_width = label[3] * img_width
            label_height = label[4] * img_height
            x_i = int(x_center - label_width/2)
            x_f = int(x_center + label_width/2)
            y_i = int(y_center - label_height/2)
            y_f = int(y_center + label_height/2)
            bbox_x = [x_i, x_f]
            bbox_y = [y_i, y_f]
            img_labels = cv.rectangle(img_labels, (bbox_x[0], bbox_y[0]), (bbox_x[1], bbox_y[1]), (0, 255, 0), 1)
    return img_labels


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))

def check_cropped(path = r"C:\Users\Utilizador\Desktop\Tese\Tarefas\Tarefa 3 - Base de Dados\Dataset\Export to Google Drive\Synthetic_Dataset_fullCropped\Cropped\Chromosomes"):
    foldernames_chromosomes = next(os.walk(path), (None, None, []))[1]
    chr_list_XX = []
    chr_list_YY = []
    for folder in foldernames_chromosomes:
        print(folder)
        folder_path = os.path.join(path, folder)
        files_chromosomes = next(os.walk(folder_path), (None, None, []))[2]
        files = []
        for filename in files_chromosomes:
            filename_split = filename.split("_")
            filename_chr = filename_split[0] + "_" + filename_split[1] + "_" + filename_split[2]
            files.append(filename_chr)
        chr_list = []
        for filename_chr in files:
            if filename_chr not in chr_list:
                chr_list.append(filename_chr)
                filename_split = filename_chr.split("_")
                if filename_split[1] == "l":
                    filename_chr_homo = filename_split[0] + "_" + "r" + "_" + filename_split[2]
                else:
                    filename_chr_homo = filename_split[0] + "_" + "l" + "_" + filename_split[2]
                if filename_chr_homo not in files:
                    if int(folder) != 24:
                        print("Delete {}".format(filename_chr))
        # print(chr_list)
        if int(folder) == 23:
            chr_list_XX = chr_list.copy()
        elif int(folder) == 24:
            chr_list_YY = chr_list.copy()
    
    # print(chr_list_YY, chr_list_XX)
    for file in chr_list_XX:
        filename_split = file.split("_")
        file_homo_r = "chYY" + "_" + "r" + "_" + filename_split[2]
        file_homo_l = "chYY" + "_" + "l" + "_" + filename_split[2]
        if (file_homo_r or file_homo_l) in chr_list_YY:
            print("Delete {}".format(file))
    # print(chr_list_YY)


def count_nr_chromosomes(path = r"C:\Users\Utilizador\Desktop\Tese\Tarefas\Tarefa 3 - Base de Dados\Dataset\Synthetic_Dataset\Synthetic_Images"):
    nr_chromosomes = 0
    nr_overlaps = 0
    nr_nucleolus = 0
    nr_other_objs = 0
    widths = []
    heights = []
    files_size = []
    foldernames_images = next(os.walk(path), (None, None, []))[1]
    for folder in foldernames_images:
        folder_path = os.path.join(path, folder)
        filenames = next(os.walk(folder_path), (None, None, []))[2]
        for filename in filenames:
            if filename.split(".")[1] == "json":
                label_path = os.path.join(folder_path, filename)
                label_data = open_json(label_path)
                nr_chromosomes = nr_chromosomes + int(label_data["Nr_Chromosomes"])
                nr_overlaps = nr_overlaps + int(label_data["Nr_Clusters"])
                nr_nucleolus = nr_nucleolus + int(label_data["Nucleolus_exists"])
                nr_other_objs = nr_other_objs + int(label_data["Nr_Random_Objects"])
                widths.append(int(label_data["imageWidth"]))
                heights.append(int(label_data["imageHeight"]))
            if filename.split(".")[1] == "tif":
                img_path = os.path.join(folder_path, filename)
                file_stats = os.stat(img_path)
                files_size.append(file_stats.st_size)
    val_widths = plt.hist(widths,'rice')
    print('Number of bins for widths: ', len(val_widths[0]))
    val_heights = plt.hist(heights,'rice')
    print('Number of bins for widths: ', len(val_heights[0]))
    
    files_size_kb = [int(i/1024) for i in files_size]

    print("Number of Chromosomes: {}\nNumber of Overlaps: {}\nNumber of Nucleolus: {}\nNumber of Random Objects: {}\n".format(\
          nr_chromosomes, nr_overlaps, nr_nucleolus, nr_other_objs))
    
    return widths, heights, files_size_kb, val_widths, val_heights
        
def count_nr_cropped_chromosomes(path = r"C:\Users\Utilizador\Desktop\Tese\Tarefas\Tarefa 3 - Base de Dados\Dataset\Synthetic_Dataset\Cropped"):
    nr_chromosomes = 0
    nr_nucleolus = 0
    nr_other_objs = 0
    foldernames = next(os.walk(path), (None, None, []))[1]
    for folder in foldernames:
        folder_path = os.path.join(path, folder)
        if folder == "Chromosomes":
            foldernames_chromosomes = next(os.walk(folder_path), (None, None, []))[1]
            for chromosome_folder_number in foldernames_chromosomes:
                chromosome_folder_number_path = os.path.join(folder_path, chromosome_folder_number)
                nr_chromosomes = nr_chromosomes + len(next(os.walk(chromosome_folder_number_path), (None, None, []))[2])
        elif folder == "Nucleolus":
            nr_nucleolus = nr_nucleolus + len(next(os.walk(folder_path), (None, None, []))[2])
        elif folder == "Other_Objects":
            nr_other_objs = nr_other_objs + len(next(os.walk(folder_path), (None, None, []))[2])
                
    print("Number of Chromosomes: {}\nNumber of Nucleolus: {}\nNumber of Random Objects: {}\n".format(\
          nr_chromosomes, nr_nucleolus, nr_other_objs))
        
        
def check_original(path = r"C:\Users\Utilizador\Desktop\Tese\Tarefas\Tarefa 3 - Base de Dados\Dataset\iCBR_Dataset\Original_Images"):
    filenames = next(os.walk(path), (None, None, []))[2]
    w_max = 0
    h_max = 0
    w_min = 10000
    h_min = 10000
    files_size = []
    for file in filenames:
        img_path = os.path.join(path, file)
        img = cv.imread(img_path)
        img_w = img.shape[1]
        img_h = img.shape[0]
        if img_w > w_max: w_max = img_w
        if img_h > h_max: h_max = img_h
        if img_w < w_min: w_min = img_w
        if img_h < h_min: h_min = img_h
        file_stats = os.stat(img_path)
        files_size.append(file_stats.st_size)
    files_size_kb = [int(i/1024) for i in files_size]
    print("Widht = [{} - {}]\nHeight = [{} - {}]\nFile size = [{} - {}]\n".format(w_min, w_max, h_min, h_max, min(files_size_kb), max(files_size_kb)))
    return files_size_kb
    
def check_YOLO_models(path = r"C:\Users\Utilizador\Desktop\Tese\Tarefas\Tarefa 5 - Resultados\Yolo_results_10000l\results", path_labels = r"C:\Users\Utilizador\Desktop\Tese\Tarefas\Tarefa 5 - Resultados\Yolo_results_10000l\content\Yolo_results_10000l\runs\detect\detection_log\labels", path_images=r"C:\Users\Utilizador\Desktop\Tese\Tarefas\Tarefa 3 - Base de Dados\Dataset\Synthetic_Dataset\iCBR_Original_Filtered\Original_Images_With_Closing\Metaphases", conf = 0.7, conf_max = 0.9):
    labels = next(os.walk(path_labels), (None, None, []))[2]
    images = next(os.walk(path_images), (None, None, []))[2]
    conf_name = int(conf*100)
    path = os.path.join(path, f"conf_{conf_name}"); os.mkdir(path)
    n_label = 0
    
    for i in range(len(labels)):
        print(images[i])
        label_path = os.path.join(path_labels, labels[i])
        img_path = os.path.join(path_images, images[i])
        
        with open(label_path) as f:
            contents = f.readlines()
        labels_yolo = []
        for line in contents:
            label = line.split(" ")
            labels_yolo.append(label)
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        # shows chromossomes and its labels in green
        img_width = img.shape[1]
        img_height = img.shape[0]
        img_labels = np.zeros((img_height, img_width, 3), dtype='uint8')
        img_labels[:,:,0] = img[:,:]
        img_labels[:,:,1] = img[:,:]
        img_labels[:,:,2] = img[:,:]
        img_label = np.zeros((img_height, img_width, 3), dtype='uint8')
        
        for label in labels_yolo:
            if float(label[5]) >= conf:
                n_label+=1
                x_center = float(label[1]) * img_width
                y_center = float(label[2]) * img_height
                label_width = float(label[3]) * img_width
                label_height = float(label[4]) * img_height
                x_i = int(x_center - label_width/2)
                x_f = int(x_center + label_width/2)
                y_i = int(y_center - label_height/2)
                y_f = int(y_center + label_height/2)
                bbox_x = [x_i, x_f]
                bbox_y = [y_i, y_f]
                r1 = randrange(1, 256)
                r2 = randrange(1, 256)
                r3 = randrange(1, 256)
                img_labels = cv.rectangle(img_labels, (bbox_x[0], bbox_y[0]), (bbox_x[1], bbox_y[1]), (r1,r2,r3), 3)
                
                if float(label[5]) < conf_max:
                    img_label[:,:,0] = img[:,:]
                    img_label[:,:,1] = img[:,:]
                    img_label[:,:,2] = img[:,:]
                    img_label = cv.rectangle(img_label, (bbox_x[0], bbox_y[0]), (bbox_x[1], bbox_y[1]), (0, 0, 255), 3)
                    cv.imshow(images[i], cv.resize(img_label, (int(img_label.shape[1] * 70 / 100), int(img_label.shape[0] * 70 / 100))))
                    cv.imshow(images[i], cv.resize(img_label, (1000 , 700)))
                    cv.waitKey(0)
        path1 = os.path.join(path, images[i])
        
        # cv.imwrite(path1, img_labels)
    print(f"\nWith confidence level at {conf_name}% , {n_label} labels were found.")
        
def split_images(path = r"C:\Users\Utilizador\Desktop\Tese\Tarefas\Tarefa 3 - Base de Dados\Dataset\Synthetic_Dataset\iCBR_Original_Filtered\Original_Images_With_Closing"):
    images = next(os.walk(path), (None, None, []))[2]
    destination_metaphases = os.path.join(path, "Metaphases")
    destination_karyograms = os.path.join(path, "Karyograms")
    for image in images:
        path_image = os.path.join(path, image)
        image_name = image.split(".")
        if image_name[0][-1] == "k":
            shutil.copy(path_image, destination_karyograms)
        else:
            shutil.copy(path_image, destination_metaphases)