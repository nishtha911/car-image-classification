import tensorflow as tf
import keras_cv
import os
import json
import matplotlib.pyplot as plt
from src import config, model_factory

def train_and_compare(train_ds, val_ds):
    results = {}
    
    # Pre-processing pipeline (resizing)
    resizing = keras_cv.layers.Resizing(
        config.IMG_SIZE[0], config.IMG_SIZE[1], bounding_box_format="xyxy"
    )
    
    train_ds = train_ds.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle and Prefetch
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    for opt_name in config.OPTIMIZERS:
        print(f"\n{'='*20}")
        print(f"Training with optimizer: {opt_name}")
        print(f"{'='*20}")
        
        model = model_factory.create_model()
        optimizer = model_factory.get_optimizer(opt_name)
        
        # RetinaNet specific compilation
        model.compile(
            classification_loss="focal",
            box_loss="smooth_l1",
            optimizer=optimizer,
            metrics=["accuracy"] # Simple accuracy for comparison, mAP is better but slower
        )
        
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.EPOCHS,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(config.MODELS_DIR, f"car_detector_{opt_name}.h5"),
                    save_best_only=True,
                    monitor="val_loss"
                ),
                tf.keras.callbacks.CSVLogger(
                    os.path.join(config.RESULTS_DIR, f"training_log_{opt_name}.csv")
                )
            ]
        )
        
        results[opt_name] = history.history
        
    # Save all results to JSON
    with open(os.path.join(config.RESULTS_DIR, "comparison_results.json"), "w") as f:
        json.dump(results, f)
        
    return results

def plot_results(results):
    plt.figure(figsize=(15, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    for opt_name, history in results.items():
        plt.plot(history['loss'], label=f'{opt_name} (train)')
        plt.plot(history['val_loss'], '--', label=f'{opt_name} (val)')
    plt.title('Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy (or mAP if we had it)
    plt.subplot(1, 2, 2)
    for opt_name, history in results.items():
        if 'accuracy' in history:
            plt.plot(history['accuracy'], label=f'{opt_name} (train)')
    plt.title('Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig(os.path.join(config.RESULTS_DIR, "comparison_plot.png"))
    plt.show()
