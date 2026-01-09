import numpy as np
from src.exception import CustomException
import sys
from src.logger import logging

class DataTransformation:
    def __init__(self, images, masks, val_split=0.2):
        self.images = images[..., np.newaxis]  # add channel
        self.masks = masks[..., np.newaxis]
        self.val_split = val_split

    def get_train_val_split(self):
        try:
            split_idx = int((1 - self.val_split) * len(self.images))
            train_images = self.images[:split_idx]
            train_masks = self.masks[:split_idx]
            val_images = self.images[split_idx:]
            val_masks = self.masks[split_idx:]
            logging.info("Data splitting is completed")
            return train_images, train_masks, val_images, val_masks
            
        except Exception as e:
            raise CustomException(e,sys)