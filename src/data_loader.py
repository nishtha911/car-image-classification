import pandas as pd
import numpy as np
import tensorflow as tf
import os
import cv2
from src import config

def load_annotations():
    df = pd.read_csv(config.ANNOTATIONS_FILE)
    
    # Group by image to handle multiple bounding boxes per image
    grouped = df.groupby('image')
    
    image_paths = []
    bboxes = []
    
    for name, group in grouped:
        img_path = os.path.join(config.TRAIN_IMAGES_DIR, name)
        if os.path.exists(img_path):
            image_paths.append(img_path)
            # KerasCV expects [xmin, ymin, xmax, ymax]
            boxes = group[['xmin', 'ymin', 'xmax', 'ymax']].values.astype(np.float32)
            # Add class ID (0 for car)
            classes = np.zeros((boxes.shape[0], 1), dtype=np.float32)
            # Combine boxes and classes if needed, or keep separate
            bboxes.append(boxes)
            
    return image_paths, bboxes

def parse_image_and_boxes(img_path, boxes):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    
    # Get original dimensions
    shape = tf.shape(image)
    orig_h = tf.cast(shape[0], tf.float32)
    orig_w = tf.cast(shape[1], tf.float32)
    
    # Normalize image to [0, 1] - EfficientNet expects [0, 255] for some versions, 
    # but KerasCV layers usually handle normalization. We'll use [0, 255] and use Rescaling later.
    image = tf.cast(image, tf.float32)
    
    # We will resize labels later using KerasCV layers for better handling
    return {"images": image, "bounding_boxes": {"boxes": boxes, "classes": tf.zeros(tf.shape(boxes)[0], dtype=tf.int32)}}

def create_dataset(image_paths, bboxes, batch_size=8, is_training=True):
    # Padding boxes to fixed size for batching
    max_boxes = max([len(b) for b in bboxes])
    
    padded_bboxes = []
    for b in bboxes:
        pad_size = max_boxes - len(b)
        if pad_size > 0:
            padded = np.pad(b, ((0, pad_size), (0, 0)), mode='constant', constant_values=-1)
        else:
            padded = b
        padded_bboxes.append(padded)
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, padded_bboxes))
    
    if is_training:
        dataset = dataset.shuffle(1000)
    
    dataset = dataset.map(parse_image_and_boxes, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Use KerasCV padding for bounding boxes
    dataset = dataset.apply(
        tf.data.experimental.dense_to_ragged_batch(batch_size)
    )
    
    return dataset

def get_data_loaders():
    image_paths, bboxes = load_annotations()
    
    # Split training/validation (80/20)
    indices = np.arange(len(image_paths))
    np.random.shuffle(indices)
    
    split = int(0.8 * len(image_paths))
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    train_paths = [image_paths[i] for i in train_indices]
    train_bboxes = [bboxes[i] for i in train_indices]
    
    val_paths = [image_paths[i] for i in val_indices]
    val_bboxes = [bboxes[i] for i in val_indices]
    
    # For now, let's keep it simple with standard batching since KerasCV layers 
    # work well with dictionaries
    
    # We need to ensure boxes are in a format KerasCV likes
    def prepare_dict(paths, boxes_list):
        # We need to handle images of different sizes if we don't resize first
        # But for training, it's better to resize to a fixed size
        dataset = tf.data.Dataset.from_generator(
            lambda: zip(paths, boxes_list),
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(None, 4), dtype=tf.float32)
            )
        )
        
        def process(path, boxes):
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.cast(image, tf.float32)
            return {"images": image, "bounding_boxes": {"boxes": boxes, "classes": tf.zeros(tf.shape(boxes)[0], dtype=tf.int32)}}

        dataset = dataset.map(process, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Resizing and Jittering can be added here
        resizing = tf.keras.layers.Resizing(config.IMG_SIZE[0], config.IMG_SIZE[1])
        
        # We'll batch first
        dataset = dataset.ragged_batch(config.BATCH_SIZE)
        
        return dataset

    train_ds = prepare_dict(train_paths, train_bboxes)
    val_ds = prepare_dict(val_paths, val_bboxes)
    
    return train_ds, val_ds
