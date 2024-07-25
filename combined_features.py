from PIL import Image
import torchvision.transforms as transforms
import torch
from transformers import ViTFeatureExtractor, ViTModel, ViTConfig, AutoConfig

# Load a sample image
# image_path = 'path_to_your_image.jpg'
# image = Image.open(image_path)

image = torch.ones([3,456,768])

# Define transforms for ResNet input
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 pixels
    #transforms.ToTensor(),  # Convert PIL Image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize tensor
])

# Apply transforms to image for ResNet
resnet_image_tensor = resnet_transform(image)

# Define feature extractor for ViT input
vit_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

# Apply feature extraction for ViT
vit_image_tensor = vit_feature_extractor(images=image, return_tensors="pt").pixel_values.squeeze(0)

# Ensure both tensors have batch dimension
resnet_image_tensor = resnet_image_tensor.unsqueeze(0)  # Add batch dimension
vit_image_tensor = vit_image_tensor.unsqueeze(0)  # Ensure ViT tensor has batch dimension
print("resnet_image_tensor", resnet_image_tensor.shape)
print("vit_image_tensor", vit_image_tensor.shape)

# Fusion Techniques
combined_tensor = vit_image_tensor + resnet_image_tensor
print("combined_tensor", combined_tensor.shape)

# Concat Version
combined_features_concat = torch.cat((resnet_image_tensor, vit_image_tensor), dim=-1)
print("combined_features_concat", combined_features_concat.shape)

from PIL import Image
import torchvision.transforms as transforms
import torch
from transformers import ViTFeatureExtractor, ViTModel, ViTConfig, AutoConfig

class FeatureFusion:
    def __init__(self, image):
        """
        Initializes the FeatureFusion class with an image tensor.
        
        Parameters:
        - image: A tensor representing the image to be processed. Expected shape is [C, H, W].
        """
        self.image = image

    def process_and_combine(self):
        """
        Processes the image for both ResNet and ViT models, and combines their features.
        
        Returns:
        - combined_features_concat: The concatenated features from both models.
        """
        # Define transforms for ResNet input
        resnet_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to 224x224 pixels
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize tensor
        ])

        # Apply transforms to image for ResNet
        resnet_image_tensor = resnet_transform(self.image)

        # Define feature extractor for ViT input
        vit_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

        # Apply feature extraction for ViT
        vit_image_tensor = vit_feature_extractor(images=self.image, return_tensors="pt").pixel_values.squeeze(0)

        # Ensure both tensors have batch dimension
        resnet_image_tensor = resnet_image_tensor.unsqueeze(0)  # Add batch dimension
        vit_image_tensor = vit_image_tensor.unsqueeze(0)  # Ensure ViT tensor has batch dimension
        
        # Fusion Techniques
        combined_tensor = vit_image_tensor + resnet_image_tensor
        print("combined_tensor", combined_tensor.shape)

        # Concat Version
        combined_features_concat = torch.cat((resnet_image_tensor, vit_image_tensor), dim=-1)
        print("combined_features_concat", combined_features_concat.shape)

        return combined_features_concat

# Example usage
image_tensor = torch.ones([3, 456, 768])
feature_fusion = FeatureFusion(image_tensor)
combined_features = feature_fusion.process_and_combine()
