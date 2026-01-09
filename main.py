from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import UNetTrainer
import matplotlib.pyplot as plt

# 1️⃣ Load data
data_ingestor = DataIngestion(
    images_path="Notebook\Adult-tooth-segmentation-dataset\Panoramic-radiography-database\images",
    masks_path="Notebook\Adult-tooth-segmentation-dataset\Panoramic-radiography-database\mask"
)
images = data_ingestor.read_images()
masks = data_ingestor.read_masks()

# 2️⃣ Transform and split
transformer = DataTransformation(images, masks)
train_images, train_masks, val_images, val_masks = transformer.get_train_val_split()

# 3️⃣ Model
trainer = UNetTrainer(input_shape=(224,224,1))

trainer.compile_model(lr=0.001)
history = trainer.train(train_images, train_masks, val_images, val_masks, epochs=4, batch_size=8)

# 4️⃣ Predictions
preds = trainer.model.predict(val_images[:5])

plt.figure(figsize=(15, 9))
for i in range(5):
    plt.subplot(3, 5, i+1)
    plt.imshow(val_images[i].squeeze(), cmap='gray')
    plt.title("Image")

    plt.subplot(3, 5, i+6)
    plt.imshow(val_masks[i].squeeze(), cmap='gray')
    plt.title("Mask")

    plt.subplot(3, 5, i+11)
    plt.imshow(preds[i].squeeze() > 0.5, cmap='gray')
    plt.title("Prediction")
plt.tight_layout()
plt.show()