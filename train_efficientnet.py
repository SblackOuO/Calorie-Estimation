import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import time
import copy

def train_model(model, criterion, optimizer, dataloaders, device, num_epochs=25, model_save_path='./models/efficientnet_b0_food.pt'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # 建立儲存模型的目錄 (如果不存在)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            if phase not in dataloaders:
                print(f"Warning: Dataloader for phase '{phase}' not found. Skipping this phase.")
                continue

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), model_save_path)
                print(f'Best val Acc: {best_acc:.4f}, Model saved to {model_save_path}')

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    # ----- 1. 設定參數 -----
    data_dir = 'data/processed/food101' # 包含 train, val, test 的父目錄
    train_subdir = 'train'
    val_subdir = 'val'   # 用於訓練過程中的驗證
    # 注意：真正的 'test' 集 (您數據的另外10%) 將用於訓練完成後的獨立評估
    model_save_name = 'efficientnet_b0_food_100_classes.pt' # 儲存模型的名稱
    num_classes = 100  # 您有100個食物類別
    batch_size = 32
    num_epochs = 25 # 可以根據需要調整
    learning_rate = 0.001

    # ----- 2. 資料轉換與載入 -----
    # 計算 ImageNet 的均值和標準差，也可以使用 EfficientNet 建議的
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224), # EfficientNet-B0 預期輸入尺寸為 224x224
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_data_path = os.path.join(data_dir, train_subdir)
    val_data_path = os.path.join(data_dir, val_subdir)

    if not os.path.exists(train_data_path):
        print(f"ERROR: Training data path does not exist: {train_data_path}")
        print("Please make sure you have created the 'train' directory inside 'data' and populated it with your training images (e.g., 80% of your data),")
        print("with each food category in its own subdirectory.")
        exit()
    if not os.path.exists(val_data_path):
        print(f"ERROR: Validation data path does not exist: {val_data_path}")
        print("Please make sure you have created the 'val' directory inside 'data' and populated it with your validation images (e.g., 10% of your data),")
        print("with each food category in its own subdirectory.")
        exit()

    image_datasets = {
        'train': datasets.ImageFolder(train_data_path, data_transforms['train']),
        'val': datasets.ImageFolder(val_data_path, data_transforms['val'])
    }
    
    # 確保 ImageFolder 確實找到了圖像
    if len(image_datasets['train']) == 0:
        print(f"ERROR: No images found in {train_data_path}. Please check the directory structure and ensure it contains images.")
        exit()
    if len(image_datasets['val']) == 0:
        print(f"ERROR: No images found in {val_data_path}. Please check the directory structure and ensure it contains images.")
        exit()

    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
        for x in ['train', 'val']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    # 檢查偵測到的類別數量是否與預期相符
    if len(class_names) != num_classes:
        print(f"Warning: Expected {num_classes} classes, but found {len(class_names)} classes in {train_data_path}.")
        print("Make sure your 'train' directory (and 'val' directory) has the correct number of subdirectories (one for each food class).")
        # 您可以選擇在這裡停止執行，或讓使用者決定是否繼續
        # exit() 
        num_classes = len(class_names) # 或者，基於找到的類別數量更新 num_classes
        print(f"Proceeding with {num_classes} classes based on the data found.")


    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ----- 3. 載入模型並修改分類頭 -----
    model_ft = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # EfficientNet 的分類器是一個 nn.Linear 層
    num_ftrs = model_ft.classifier[1].in_features
    model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)

    model_ft = model_ft.to(device)

    # ----- 4. 定義損失函數和優化器 -----
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate)

    # ----- 5. 訓練模型 -----
    print("Starting training...")
    trained_model = train_model(model_ft, criterion, optimizer_ft, dataloaders, device,
                                num_epochs=num_epochs, 
                                model_save_path=os.path.join('models', model_save_name))
    
    print(f"Training finished. Best model saved to {os.path.join('models', model_save_name)}")

    # ----- 6. (可選) 儲存類別名稱 -----
    # 儲存 class_names 以便後續推論時使用
    try:
        with open(os.path.join('models', model_save_name + '.classes.txt'), 'w') as f:
            for item in class_names:
                f.write("%s\n" % item)
        print(f"Class names saved to {os.path.join('models', model_save_name + '.classes.txt')}")
    except Exception as e:
        print(f"Error saving class names: {e}") 