import numpy as np
import cv2
import tensorflow as tf

IMG_SIZE = 224

class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, dataframe, batch_size, augment=False):
        self.df = dataframe
        self.batch_size = batch_size
        self.augment = augment
        
    def __len__(self):
        return len(self.df) // self.batch_size
    
    def __getitem__(self, index):
        
        batch = self.df.iloc[index*self.batch_size:(index+1)*self.batch_size]
        
        images = []
        bboxes = []
        
        for _, row in batch.iterrows():
            
            img = cv2.imread(row.image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            h, w, _ = img.shape
            
            xmin = row.xmin / w
            ymin = row.ymin / h
            xmax = row.xmax / w
            ymax = row.ymax / h
            
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            
            if self.augment:
                if np.random.rand() > 0.5:
                    img = np.fliplr(img)
                    xmin, xmax = 1-xmax, 1-xmin
            
            bbox = [xmin, ymin, xmax, ymax]
            
            images.append(img)
            bboxes.append(bbox)
        
        return np.array(images), np.array(bboxes)