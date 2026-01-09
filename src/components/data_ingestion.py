import os
import cv2
import numpy as np
from pathlib import Path
from src.exception import CustomException
import sys
from src.logger import logging

class DataIngestion:
    def __init__(self, images_path, masks_path, img_size=(224,224)):
        self.images_path = Path(images_path)
        self.masks_path = Path(masks_path)
        self.img_size = img_size

    def read_images(self):
      
        try:
            images = []
            for filename in sorted(os.listdir(self.images_path)):
                img = cv2.imread(str(self.images_path / filename), 0)
                img = cv2.resize(img, self.img_size)
                img = img / 255.0
                images.append(img)
            logging.info("Images reading is completed")
            return np.array(images)
            
        except Exception as e:
            raise CustomException(e,sys)
            

    def read_masks(self):
        try:
            masks = []
            for filename in sorted(os.listdir(self.masks_path)):
                mask = cv2.imread(str(self.masks_path / filename), 0)
                mask = cv2.resize(mask, self.img_size)
                mask = mask / 255.0
                mask = (mask > 0.5).astype(np.float32)
                masks.append(mask)
            logging.info("masks Reading is completed")
            return np.array(masks)
            
        except Exception as e:
            raise CustomException(e,sys)
            