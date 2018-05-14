"""
Utilities for Painting classification
"""
import numpy as np
import tensorflow as tf
from lxml import etree as ET

VGG_MEAN = tf.constant([123.68, 116.78, 103.94])

# def preprocess_image(file_name, label):
#     """
#     Preprocess by parsing, decoding jpeg, resize and normalize RGB channel
#     """
#     image = tf.image.decode_jpeg(tf.read_file(file_name), channels=3)
#     image = tf.cast(image, tf.float32)
    
#     #resize image with smaller side to 256
#     new_length = tf.constant(256)
    
#     if tf.shape(image)[0] < tf.shape(image)[1]: #if height is smaller
#         resized_image = tf.image.resize_images(image, [256, 256])
#         crop = tf.random_crop(resized_image, [224, 224, 3]) #crop [224,224]
#         vgg_means = tf.reshape(VGG_MEAN, [1,1,3])

#         #vgg trained without normalization
#         return (crop - vgg_means), label #normalize by subtracting
#     else:
#         resized_image = tf.image.resize_images(image, [256, 256])
#         crop = tf.random_crop(resized_image, [224, 224, 3]) #crop [224,224]
#         vgg_means = tf.reshape(VGG_MEAN, [1,1,3])
#         return (crop - vgg_means), label

def preprocess_image(file_name, label):
    """
    Preprocess by parsing, decoding jpeg, resize and normalize RGB channel
    """
    image = tf.image.decode_jpeg(tf.read_file(file_name), channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    
    h, w = tf.to_float(tf.shape(image)[0]), tf.to_float(tf.shape(image)[1])
    #resize image with smaller side to 256
    smallest_side = 225.0
    
    scale = tf.cond(tf.greater(h, w),
                lambda: smallest_side / w,
                lambda: smallest_side / h)
    
    new_height = tf.to_int32(h * scale)
    new_width = tf.to_int32(w * scale)
    
    resized_image = tf.image.resize_images(image, [new_height, new_width])
    vgg_means = tf.reshape(VGG_MEAN, [1,1,3])
    
    #cropping
    crop = tf.random_crop(resized_image, [224,224,3])
    #vgg trained without normalization
    return crop, label #normalize by subtracting

def preprocess_image_flip(file_name, label, flip_prob=0.3):
    """
    Preprocess by parsing, decoding jpeg, resize and normalize RGB channel
    """
    image = tf.image.decode_jpeg(tf.read_file(file_name), channels=3)
    image = tf.cast(image, tf.float32)
    
    if np.random.rand() < flip_prob:
        image = tf.image.flip_left_right(image)
        
    image = tf.image.per_image_standardization(image)
    
    h, w = tf.to_float(tf.shape(image)[0]), tf.to_float(tf.shape(image)[1])
    #resize image with smaller side to 256
    smallest_side = 225.0
    
    scale = tf.cond(tf.greater(h, w),
                lambda: smallest_side / w,
                lambda: smallest_side / h)
    
    new_height = tf.to_int32(h * scale)
    new_width = tf.to_int32(w * scale)
    
    resized_image = tf.image.resize_images(image, [new_height, new_width])
    vgg_means = tf.reshape(VGG_MEAN, [1,1,3])
    
    #cropping
    crop = tf.random_crop(resized_image, [224,224,3])
    #vgg trained without normalization
    return crop, label #normalize by subtracting
    
    
def val_preprocess_image(file_name, label):
    image = tf.image.decode_jpeg(tf.read_file(file_name), channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    crop = tf.image.resize_image_with_crop_or_pad(image, 224, 224) 
    vgg_means = tf.reshape(VGG_MEAN, [1, 1, 3])                                
    return crop, label
    
def get_labels(info, label='artist'):
    """
    Returns list of labels from extracted info. dataframe
    """
    unique_labels = list(set(info[label]))
    label_map = dict(zip(unique_labels, range(len(unique_labels))))
    return [label_map[l] for l in list(info[label])], label_map

def extract_xml_info(root):
    """
    Extract info. from .xml file. 
    """
    info = root[1][0]
    title = None
    artist = None
    medium = None
    period = None
    for i in info:
        if i.tag == 'dc:title':
            title = i.text
        if i.tag == 'dc:creator':
            artist = i.text
        if i.tag == 'dc:type':
            medium = i.text
        if i.tag == 'dc:coverage':
            period = i.text
    return (title, artist, medium, period)