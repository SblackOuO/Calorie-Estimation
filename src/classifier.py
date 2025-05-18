import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class FoodClassifier:
    """
    Food classification wrapper that adapts to number of classes automatically.
    """
    def __init__(self, model_path: str, train_dir: str = None):
        # Determine device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Determine number of classes
        if train_dir and os.path.isdir(train_dir):
            self.class_names = sorted([d for d in os.listdir(train_dir)
                                       if os.path.isdir(os.path.join(train_dir, d))])
        else:
            # Fallback: infer from checkpoint
            state = torch.load(model_path, map_location='cpu')
            num_classes = state['classifier.1.weight'].shape[0]
            self.class_names = [str(i) for i in range(num_classes)]
        num_classes = len(self.class_names)
        # Initialize model with proper head
        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)
        # Load checkpoint
        state_dict = torch.load(model_path, map_location=self.device)
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError:
            # Allow missing keys for flexibility
            self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device).eval()

    def predict(self, pil_img):
        """
        Predict the class name for a PIL image.
        """
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        x = preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            idx = logits.argmax(dim=1).item()
        return self.class_names[idx]
