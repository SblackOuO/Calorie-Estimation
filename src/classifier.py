import os
import torch
import torch.nn as nn
from torchvision import transforms
#from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class FoodClassifier:
    """
    Food classification wrapper that loads a fine-tuned EfficientNet-B0.
    """
    def __init__(self, model_path: str, train_dir: str):
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load class names from train_dir
        self.class_names = sorted([
            d for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        ])
        # IMPORTANT: num_classes should match the loaded model's output.
        # 'efficientnet_b0_food_v2_100_classes.pt' was found to be for 101 classes.
        num_classes = 101 # <--- 確保這是101以匹配模型權重

        # Build EfficientNet-B0 model
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Modify the classifier head for EfficientNet
        if isinstance(self.model.classifier, nn.Sequential) and len(self.model.classifier) > 0:
            try:
                num_ftrs = self.model.classifier[1].in_features
                self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            except (IndexError, AttributeError) as e:
                print(f"無法自動修改EfficientNet分類頭的最後一層：{e}")
                print("將嘗試直接替換整個 model.classifier。")
                if hasattr(self.model.classifier, 'in_features'): 
                     num_ftrs = self.model.classifier.in_features
                     self.model.classifier = nn.Linear(num_ftrs, num_classes)
                else: 
                    print("錯誤：無法確定如何修改EfficientNet的分類頭。請檢查模型結構。")
                    raise AttributeError("無法找到EfficientNet分類頭的in_features。")
        else:
            try:
                num_ftrs = self.model.classifier.in_features
                self.model.classifier = nn.Linear(num_ftrs, num_classes)
            except AttributeError:
                print("錯誤：無法修改EfficientNet的分類頭。model.classifier 結構未知。")
                raise AttributeError("EfficientNet的model.classifier不符合預期結構。")

        # Load fine-tuned weights
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=True)
        except FileNotFoundError:
            print(f"錯誤: 模型檔案 {model_path} 未找到。")
            raise
        except RuntimeError as e:
            print(f"錯誤: 載入模型權重時發生 RuntimeError (可能由於類別數量或鍵名不匹配): {e}")
            print(f"FoodClassifier期望一個針對 {num_classes} 個類別訓練的EfficientNet-B0模型。")
            print(f"請確認 {model_path} 指向正確的模型檔案，且其鍵名與當前模型結構匹配。")
            raise

        # Move to device & set eval mode
        self.model.to(self.device)
        self.model.eval()

        # Define preprocessing: same as validation
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)), # EfficientNet-B0 expects 224x224
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]) # Standard ImageNet normalization
        ])

    def predict(self, pil_img):
        """
        Predict the class label for a single PIL image.
        Returns the class name.
        """
        if not self.class_names:
            print("錯誤: class_names 未被初始化，無法預測。")
            return "Error: class_names not loaded"
            
        img = pil_img.convert('RGB')
        x = self.val_transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
        idx = logits.argmax(dim=1).item()
        
        if idx >= len(self.class_names):
            print(f"警告: 模型預測索引 {idx} 超出 class_names 列表範圍 (大小: {len(self.class_names)}).")
            # Try to get the number of output classes from the model's classifier
            model_output_classes = -1
            if isinstance(self.model.classifier, nn.Sequential) and len(self.model.classifier) > 1 and isinstance(self.model.classifier[1], nn.Linear):
                model_output_classes = self.model.classifier[1].out_features
            elif isinstance(self.model.classifier, nn.Linear):
                model_output_classes = self.model.classifier.out_features
            print(f"這可能表示 train_dir 中的類別數量 ({len(self.class_names)}) 與模型輸出的類別數量 ({model_output_classes}) 不完全匹配。")
            return f"PredictionIndexOutOfRange (idx: {idx}, names_len: {len(self.class_names)}, model_outputs: {model_output_classes})"
        return self.class_names[idx]