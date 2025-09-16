import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
from sklearn.preprocessing import normalize
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self, modelname="resnet34"):
        logger.info(f"Initializing FeatureExtractor with model={modelname}")

        try:
            # Load pretrained ResNet34 from torchvision
            base_model = models.resnet34(weights="IMAGENET1K_V1")
            logger.debug("ResNet34 pretrained weights loaded successfully.")
        except Exception as e:
            logger.exception("Failed to load ResNet34 model.")
            raise

        # Remove final classification layer (fc) → keep as embedding extractor
        self.model = torch.nn.Sequential(*list(base_model.children())[:-1])
        self.model.eval()
        logger.debug("Final classification layer removed, model ready as feature extractor.")

        # Image preprocessing (ImageNet normalization)
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        logger.debug("Preprocessing pipeline set up (resize → tensor → normalize).")

        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Feature extractor running on device: {self.device}")

    def __call__(self, imagepath):
        logger.info(f"Extracting features from image: {imagepath}")

        try:
            img = Image.open(imagepath).convert("RGB")
            logger.debug("Image opened and converted to RGB successfully.")
        except (FileNotFoundError, UnidentifiedImageError) as e:
            logger.exception(f"Failed to open image {imagepath}")
            raise

        try:
            tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            logger.debug(f"Image preprocessed and moved to device {self.device}. Shape: {tensor.shape}")
        except Exception as e:
            logger.exception("Error during preprocessing.")
            raise

        try:
            with torch.no_grad():
                vec = self.model(tensor).squeeze().cpu().numpy()
            logger.debug(f"Raw feature vector extracted. Shape: {vec.shape}")
        except Exception as e:
            logger.exception("Error during model forward pass.")
            raise

        try:
            norm_vec = normalize(vec.reshape(1, -1), norm="l2").flatten()
            logger.info(f"Feature vector extracted successfully. Vector length: {len(norm_vec)}")
            return norm_vec
        except Exception as e:
            logger.exception("Error during normalization of feature vector.")
            raise

# Cache the extractor instance to avoid reloading the model multiple times
@lru_cache(maxsize=1)
def get_extractor():
    logger.info("Retrieving (cached) FeatureExtractor instance.")
    return FeatureExtractor("resnet34")
