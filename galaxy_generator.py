import sys
sys.path.insert(0, 'C:/Users/Max Power/OneDrive/ponte/programmi/python/progetto2/AJ_lib')
# sys.path.insert(0, 'C:/Users/ajacassi/OneDrive/ponte/programmi/python/progetto2/AJ_lib')
from AJ_draw import disegna as ds
from PIL import Image
from matplotlib import pyplot
import random
import numpy as np
from astropy.io import fits

import plotly.express as px
import pickle
from os import listdir

from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array

import streamlit as st

def plot_galaxies_xo_box(image):
    pyplot.imshow(image)
    pyplot.show()

def plot_galaxies(image,box_dict):
    # print(box_dict)
    ds().nuova_fig(1)
    for i in range(len(box_dict)):
        # x_min = box_dict[i,0]
        # y_min = box_dict[i,1]
        # x_max = box_dict[i,2]
        # y_max = box_dict[i,3]

        y_min = box_dict[i,0]
        x_min = box_dict[i,1]
        y_max = box_dict[i,2]
        x_max = box_dict[i,3]
        ds().dati(x=[x_min,x_max,x_max,x_min,x_min], y=[y_min,y_min,y_max,y_max,y_min], larghezza_riga =5, colore='red', layer =2)
    y = np.linspace(0, image.shape[0], image.shape[0])
    x = np.linspace(0, image.shape[1], image.shape[1])
    ds().dati(x = x, y = y, z = image, scat_plot ='cmap')
    # ds().porta_a_finestra()
    st.pyplot()
    # pyplot.imshow(image)
    # pyplot.show()

def data_set_generator(image):
    ''' divide the image in sub images'''
    sub_set = 40
    y_shape = image.shape[0]//sub_set - 1
    x_shape = image.shape[1]//sub_set
    images = np.zeros((sub_set**2, y_shape, x_shape))

    n_image = 0
    for n_image_x in range(sub_set):
        for n_image_y in range(sub_set):
            images[n_image] = image[n_image_y*y_shape : (n_image_y+1)*y_shape, n_image_x*x_shape : (n_image_x+1)*x_shape]
            n_image = n_image + 1
    # images[0] = image[0 : y_shape, 0 :x_shape]
    return images

def crop_circle(image, centre, crop_rad, mean):
    ''' replace a galaxy with the back ground'''
    for i in range(centre[0] - crop_rad, centre[0] + crop_rad):
        if i >= 0 and i < image.shape[0]:
            for j in range(centre[1] - crop_rad, centre[1] + crop_rad):
                if j >= 0 and j < image.shape[1]:
                    dist_centro = np.sqrt((centre[0] - i)**2 + (centre[1] - j)**2)
                    if dist_centro < crop_rad:
                        image[i][j] = mean
    return image

def galaxy_finder(image, image_original):
    ''' finds the galaxy in the image related to the pixel with the maximum intesity'''
    mean = image_original.mean()
    crop_rad = 9
    ind = np.unravel_index(np.argmax(image, axis=None), image.shape)
    # st.write('coord', ind)

    galaxy = np.zeros((crop_rad*2, crop_rad*2))
    galaxy_temp = image_original[ind[0] - crop_rad : ind[0] + crop_rad, ind[1] - crop_rad : ind[1] + crop_rad]
    galaxy[:galaxy_temp.shape[0], :galaxy_temp.shape[1]] = galaxy_temp

###########################
    # tracciare la posizione anche
##########################

    # image[ind[0] - galaxy_temp.shape[0]//2 : ind[0] + galaxy_temp.shape[0]//2, ind[1] - galaxy_temp.shape[0]//2 : ind[1] + galaxy_temp.shape[0]//2] = mean
    image = crop_circle(image, ind, crop_rad, mean)
    return galaxy, image

def galaxy_separator(image):
    ''' generates a matrix with all the galaxies as single images '''

    image_galaxy = image.copy()
    single_galaxy_new, image_galaxy = galaxy_finder(image_galaxy, image)
    # plot_galaxies(single_galaxy_new)

    while image_galaxy.max() >= 1.1*image.mean():
    # for i in range(50):
        # st.write(image_galaxy.max(), image.mean(), image_galaxy.mean())
        single_galaxy, image_galaxy = galaxy_finder(image_galaxy, image)
        # plot_galaxies(single_galaxy)
        single_galaxy_new = np.dstack((single_galaxy_new, single_galaxy))
    # st.write(single_galaxy_new.shape)
    # plot_galaxies(image_galaxy)

def galaxy_generator(xc, yc, amp, theta, sigma_x, sigma_y, x_frame, y_frame, verbose = 0):
    def gaussian_2d(x, y, amp, xc, yc, sigma_x, sigma_y, theta):
        a = np.cos(theta)**2/(2*sigma_x**2) + np.sin(theta)**2/(2*sigma_y**2);
        b = -np.sin(2*theta)/(4*sigma_x**2) + np.sin(2*theta)/(4*sigma_y**2);
        c = np.sin(theta)**2/(2*sigma_x**2) + np.cos(theta)**2/(2*sigma_y**2);
        gauss = amp*np.exp( - (a*(x-xc)**2 + 2*b*(x-xc)*(y-yc) + c*(y-yc)**2));
        return np.ravel(gauss)
    x = np.linspace(0, x_frame, x_frame, endpoint=False)
    y = np.linspace(0, y_frame, y_frame, endpoint=False)
    intensity_gauss = np.zeros((len(x),len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            intensity_gauss[i,j] = gaussian_2d(x[i], y[j], amp, xc, yc, sigma_x, sigma_y, theta)
    # if verbose == 1:
    #     plot_galaxies(intensity_gauss)
    return intensity_gauss

def galaxy_map_generator(num_galax, selected_image, verbose = 0):
    max = selected_image.max()
    x_vet = []
    y_vet = []
    theta = []
    for num in range(num_galax):
        x_vet.append(random.randrange(selected_image.shape[0]))
        y_vet.append(random.randrange(selected_image.shape[1]))
        theta.append(random.uniform(0,2*3.14))

    amp = np.random.normal(max,0.2,num_galax)
    sigma_x = np.random.normal(3, 0.7, num_galax)
    sigma_y = np.random.normal(3, 0.7, num_galax)

    empty_map = np.random.normal(0,0.25*max,selected_image.shape[0]*selected_image.shape[1])
    empty_map +=  np.abs(empty_map.min())
    empty_map = empty_map.reshape(selected_image.shape[0], selected_image.shape[1])

    type_of_galaxy = np.ones(num_galax)

    box = np.zeros((num_galax, 4))
    for i in range(num_galax):
        box[i] = [x_vet[i]-10, y_vet[i]-10, x_vet[i]+10, y_vet[i]+10]
        if box[i,3]>64: box[i,3] = 64
        if box[i,2]>114: box[i,2] = 114
        for j in range(4):
            if box[i,j] < 0: box[i,j] = 0
        fake_galaxy = galaxy_generator(x_vet[i], y_vet[i], amp[i], theta[i], sigma_x[i], sigma_y[i], selected_image.shape[0], selected_image.shape[1])
        if random.random()>0.98 and amp[i] > max:
            fake_galaxy += galaxy_generator(x_vet[i], y_vet[i], max*4, 3.14/2, 0.7, 6, selected_image.shape[0], selected_image.shape[1])
        empty_map += fake_galaxy
        if sigma_x[i] >= sigma_y[i]-0.2 and sigma_x[i] <= sigma_y[i]+0.2:
            type_of_galaxy[i] = 2
    # if verbose == 1:
    #     plot_galaxies(empty_map)
    return empty_map, box, type_of_galaxy

def save_obj(obj, name ):
    with open('fake_galaxy/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj():
    with open('fake_galaxy/dictionary_boxes.pkl', 'rb') as f:
        return pickle.load(f)

from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString

def save_annotation(obj, name, width, height, type_of_galaxy):
    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'fake_galaxy'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = name+'.jpg'

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(width)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(height)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    for i in range(len(obj)):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        if type_of_galaxy[i] == 1:
            node_name.text = 'elliptic'
        else:
            node_name.text = 'circular'
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(obj[i,0])
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(obj[i,1])
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(obj[i,2])
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(obj[i,3])

    xml = tostring(node_root, pretty_print=True) #Formatted display, the newline of the newline
    doc = parseString(xml)
    f = open('./fake_galaxy/annots/'+name+".xml", "w")
    doc.writexml(f)
    f.close()



#  █████  ██████  ██████  ██
# ██   ██ ██   ██ ██   ██ ██
# ███████ ██████  ██████  ██
# ██   ██ ██      ██   ██ ██
# ██   ██ ██      ██   ██ ██


hdul = fits.open('A1_mosaic.fits')
image = hdul[0].data
hdul.close()
# plot_galaxies_xo_box(image)
images = data_set_generator(image)
max_max = images.max()
selected_image = images[1481]
# plot_galaxies_xo_box(selected_image)


# ███████  █████  ██    ██ ███████      ██████  ██████  ██  ██████  ██ ███    ██  █████  ██      ███████
# ██      ██   ██ ██    ██ ██          ██    ██ ██   ██ ██ ██       ██ ████   ██ ██   ██ ██      ██
# ███████ ███████ ██    ██ █████       ██    ██ ██████  ██ ██   ███ ██ ██ ██  ██ ███████ ██      ███████
#      ██ ██   ██  ██  ██  ██          ██    ██ ██   ██ ██ ██    ██ ██ ██  ██ ██ ██   ██ ██           ██
# ███████ ██   ██   ████   ███████      ██████  ██   ██ ██  ██████  ██ ██   ████ ██   ██ ███████ ███████

# selected_image = selected_image*max_max
# selected_image = selected_image.reshape(selected_image.shape[0], selected_image.shape[1], 1)
# save_img('fake_galaxy/original/'+str(1562).zfill(4)+'.jpg', selected_image)

# for i in range(images.shape[0]):
# for i in range(20):
#     selected_image = images[i]
#     plot_galaxies_xo_box(selected_image)
    # selected_image = selected_image.reshape(selected_image.shape[0], selected_image.shape[1], 1)
    # save_img('fake_galaxy/original/'+str(i).zfill(4)+'.jpg', selected_image)


#  ██████ ██████  ███████  █████
# ██      ██   ██ ██      ██   ██
# ██      ██████  █████   ███████
# ██      ██   ██ ██      ██   ██
#  ██████ ██   ██ ███████ ██   ██

# #
# num_maps = 250
# box_dict = dict()
# for i in range(num_maps):
#     num_galax_in_place = random.randrange(3,12)
#     fake_galax, box_dict[i], type_of_galaxy = galaxy_map_generator(num_galax_in_place, selected_image, verbose =1)
#     plot_galaxies(fake_galax, box_dict[i])
#     fake_galax = fake_galax.reshape(fake_galax.shape[0], fake_galax.shape[1], 1)
#     save_img('fake_galaxy/images/'+str(i).zfill(4)+'.jpg', fake_galax)
#     save_annotation(box_dict[i], str(i).zfill(4), fake_galax.shape[0], fake_galax.shape[1], type_of_galaxy)
#     print(i)
### save_obj(box_dict, 'dictionary_boxes')


#  ██████  ██████  ███    ██ ████████ ██████   ██████  ██      ██       █████
# ██      ██    ██ ████   ██    ██    ██   ██ ██    ██ ██      ██      ██   ██
# ██      ██    ██ ██ ██  ██    ██    ██████  ██    ██ ██      ██      ███████
# ██      ██    ██ ██  ██ ██    ██    ██   ██ ██    ██ ██      ██      ██   ██
#  ██████  ██████  ██   ████    ██    ██   ██  ██████  ███████ ███████ ██   ██

#
# images_dir = 'fake_galaxy' + '/images/'
# annotations_dir = 'fake_galaxy'
# is_train = True
# img = dict()
# i = 0
# for filename in listdir(images_dir):
#     image_id = filename[:-4]
#     if is_train and int(image_id) >= 200:
#         continue
#     if not is_train and int(image_id) < 200:
#         continue
#     img_path = images_dir + filename
#     img[i] = img_to_array(load_img(img_path))
#     i += 1
#
# ann_path = annotations_dir + 'dictionary_boxes.pkl'
# dict_boxes = load_obj()
#
# for i in range(len(img)):
#     plot_galaxies(img[i][:,:,0],dict_boxes[i])
#     print(i)


# images_dir = 'fake_galaxy' + '/original/'
# img = dict()
# i = 0
# for filename in listdir(images_dir):
#     image_id = filename[:-4]
#     img_path = images_dir + filename
#     img[i] = img_to_array(load_img(img_path))
#     i += 1
#
# for i in range(len(img)):
#     plot_galaxies_xo_box(img[i][:,:,0])
#     print(i)
