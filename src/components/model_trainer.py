import tensorflow as tf
from src.exception import CustomException
import sys
from src.logger import logging

class UNetTrainer:
    def __init__(self, input_shape=(224,224,1), base_filters=32):
        self.input_shape = input_shape
        self.base_filters = base_filters
        self.model = self.unet()

    def convolution(inputs, filter, padding, strides, kernel, activation, conv_type):
        try:
            x = inputs
            x = tf.keras.layers.Conv2D(filter, kernel_size = kernel, padding = padding,
                                strides = strides)(x)
            x = tf.keras.layers.GroupNormalization(groups = filter)(x)
            if conv_type == 'decoder':
                x = tf.keras.layers.Activation(activation)(x)
                x = tf.keras.layers.Conv2D(filter*2, kernel_size = kernel, padding = padding, strides = strides)(x)
                x = tf.keras.layers.GroupNormalization(groups = filter*2)(x)
                x = tf.keras.layers.Activation(activation)(x)
                x = tf.keras.layers.Conv2D(filter, kernel_size = kernel, padding = padding, strides = strides)(x)
                x = tf.keras.layers.GroupNormalization(groups = filter)(x)
            x = tf.keras.layers.average([x, tf.keras.layers.Conv2D(filter, kernel_size = 1, padding = 'same',
                                                strides = 1)(inputs)])
            x = tf.keras.layers.Activation(activation)(x)
            logging.info("Convolution layers have been successfully built.")
            return x
            
        except Exception as e:
            raise CustomException(e,sys)

    def encoder(input, filter, padding, strides, kernel, activation):
        try:
            x = input
            x = UNetTrainer.convolution(x, filter, padding, strides, kernel, activation, 'encoder')
            downsample = tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
            logging.info("Encoder building is completed")
            return downsample, x
            
        except Exception as e:
            raise CustomException(e,sys)

    def decoder(input, filter, skip, padding, strides, kernel, activation):
        try:
            x = input
            x = tf.keras.layers.Conv2DTranspose(filter, kernel_size = kernel, padding = padding,
                                        strides = 2, activation = activation)(x)
            x = tf.keras.layers.average([x, skip])
            x = UNetTrainer.convolution(x, filter, padding, strides, kernel, activation, 'decoder')
            logging.info("Decoder building is completed")
            return x
            
        except Exception as e:
            raise CustomException(e,sys)

    def unet(self):
        try:
            # 1️⃣ Input layer
            inputs = tf.keras.layers.Input(shape=self.input_shape)

            # 2️⃣ Encoder (DOWN path)
            e1, s1 = UNetTrainer.encoder(inputs, self.base_filters, 'same', 1, 3, 'relu')
            e2, s2 = UNetTrainer.encoder(e1, self.base_filters*2, 'same', 1, 3, 'relu')
            e3, s3 = UNetTrainer.encoder(e2, self.base_filters*4, 'same', 1, 3, 'relu')
            e4, s4 = UNetTrainer.encoder(e3, self.base_filters*8, 'same', 1, 3, 'relu')

            # 3️⃣ Bottleneck
            b = UNetTrainer.convolution(e4, self.base_filters*16, 'same', 1, 3, 'relu', 'encoder')

            # 4️⃣ Decoder (UP path)
            d1 = UNetTrainer.decoder(b, self.base_filters*8, s4, 'same', 1, 3, 'relu')
            d2 = UNetTrainer.decoder(d1, self.base_filters*4, s3, 'same', 1, 3, 'relu')
            d3 = UNetTrainer.decoder(d2, self.base_filters*2, s2, 'same', 1, 3, 'relu')
            d4 = UNetTrainer.decoder(d3, self.base_filters, s1, 'same', 1, 3, 'relu')

            # 5️⃣ Output
            outputs = tf.keras.layers.Conv2D(
                1, kernel_size=1, activation='sigmoid'
            )(d4)

            # 6️⃣ Model
            model = tf.keras.models.Model(inputs, outputs, name="UNet")
            logging.info("Unet model building is completed")
            return model
            
        except Exception as e:
            raise CustomException(e,sys)

    def compile_model(self, lr=0.001):
        try:
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            logging.info("Model compiling is completed")
        except Exception as e:
            raise CustomException(e,sys)

            

    def train(self, train_images, train_masks, val_images, val_masks, epochs=10, batch_size=8):
        try:
            history = self.model.fit(
                train_images, train_masks,
                validation_data=(val_images, val_masks),
                epochs=epochs,
                batch_size=batch_size
            )
            logging.info("Model training is completed")
            return history
            
        except Exception as e:
            raise CustomException(e,sys)