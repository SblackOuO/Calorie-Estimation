import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class FoodClassifier:
    """
    Food classification wrapper that loads a fine-tuned MobileNetV2 on Food-101.
    """
    def __init__(self, model_path: str, train_dir: str):
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load class names
        self.class_names = sorted([
            d for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        ])
        num_classes = len(self.class_names)-1

        # Build model: same head as in training
        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        # Replace classifier head with Dropout + Linear(num_classes)
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.last_channel, num_classes)
        )

        # Load fine-tuned weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=True)

        # Move to device & set eval mode
        self.model.to(self.device)
        self.model.eval()

        # Define preprocessing: same as validation
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def predict(self, pil_img):
        """
        Predict the class label for a single PIL image.
        Returns the class name.
        """
        # Ensure RGB
        img = pil_img.convert('RGB')
        # Apply preprocessing
        x = self.val_transform(img).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(x)
        idx = logits.argmax(dim=1).item()
        return self.class_names[idx]