import keras_cv
import tensorflow as tf
from src import config

def create_model(backbone_name="efficientnet_v2_b0"):
    # Create the backbone
    backbone = keras_cv.models.Backbone.from_preset(backbone_name, load_weights=True)
    
    # Create RetinaNet with the backbone
    model = keras_cv.models.RetinaNet(
        num_classes=config.NUM_CLASSES,
        bounding_box_format="xyxy",
        backbone=backbone,
    )
    
    return model

def get_optimizer(name, lr=config.LEARNING_RATE):
    name = name.lower()
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    elif name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    elif name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr)
    elif name == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=lr)
    else:
        raise ValueError(f"Unknown optimizer: {name}")
