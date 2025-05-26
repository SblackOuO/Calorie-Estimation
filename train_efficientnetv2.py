import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import time
import copy
from tqdm import tqdm # 引入 tqdm

def train_efficientnet_model(data_dir, train_subdir, val_subdir, num_classes, 
                             model_save_name='efficientnet_b0_food_100_classes.pt', 
                             batch_size=128, num_epochs=30, learning_rate=0.001, 
                             patience=5, label_smoothing=0.1, weight_decay=1e-4):
    
    # ----- 裝置設定 -----
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        # MPS 的 autocast 通常不需要 GradScaler
        # scaler = torch.cuda.amp.GradScaler(enabled=False) if device.type == 'mps' else None # 這樣寫mps會報錯
        scaler = None # MPS 不使用 GradScaler
        use_autocast = True
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        scaler = torch.cuda.amp.GradScaler()
        use_autocast = True
    else:
        device = torch.device("cpu")
        scaler = None
        use_autocast = False # CPU 通常不從 autocast 獲益，除非有特定 Intel 硬體支援
    print(f"Using device: {device}")

    # ----- 資料轉換與載入 -----
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0) # 調整 p 和 scale
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
        exit()
    if not os.path.exists(val_data_path):
        print(f"ERROR: Validation data path does not exist: {val_data_path}")
        exit()

    image_datasets = {
        'train': datasets.ImageFolder(train_data_path, data_transforms['train']),
        'val': datasets.ImageFolder(val_data_path, data_transforms['val'])
    }
    
    if len(image_datasets['train']) == 0:
        print(f"ERROR: No images found in {train_data_path}.")
        exit()
    if len(image_datasets['val']) == 0:
        print(f"ERROR: No images found in {val_data_path}.")
        exit()

    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x=='train'), num_workers=4)
        for x in ['train', 'val']
    }
    class_names = image_datasets['train'].classes
    actual_num_classes = len(class_names)

    if actual_num_classes != num_classes:
        print(f"Warning: Expected {num_classes} classes based on parameter, but found {actual_num_classes} classes in {train_data_path}.")
        print(f"Proceeding with {actual_num_classes} classes based on the data found.")
        num_classes = actual_num_classes # 更新 num_classes 以匹配實際數據

    # ----- 模型設定 -----
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # ----- 損失函數, 優化器, 排程器 -----
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # ----- 訓練迴圈 -----
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    no_improve_epochs = 0
    model_save_full_path = os.path.join('models', model_save_name)
    os.makedirs(os.path.dirname(model_save_full_path), exist_ok=True)

    print(f'>>> Start training EfficientNet-B0 on {device}')
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            dataset_size = len(dataloaders[phase].dataset)

            progress_bar = tqdm(dataloaders[phase], desc=f'{phase.capitalize():<10}', unit='batch')

            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                
                # 混合精度
                if use_autocast:
                    with torch.autocast(device_type=device.type, enabled=use_autocast):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    if phase == 'train' and scaler: # scaler 僅用於 CUDA
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    elif phase == 'train': # MPS 或 CPU (autocast 但無 scaler)
                        loss.backward()
                        optimizer.step()
                else: # 不使用 autocast (例如，明確禁用或 CPU 且無特定支援)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'train':
                    progress_bar.set_postfix(loss=loss.item(), acc=running_corrects.double() / (progress_bar.n + 1) / batch_size if progress_bar.n > 0 else 0.0)

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print(f'{phase.capitalize():<10} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                scheduler.step() # 通常在驗證階段後更新排程器
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), model_save_full_path)
                    print(f'    Saved best model (val_acc={best_acc:.4f}) to {model_save_full_path}')
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    if no_improve_epochs >= patience:
                        print(f'No improvement for {patience} epochs. Early stopping.')
                        time_elapsed = time.time() - since
                        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s (Early stopped)')
                        print(f'Best val Acc: {best_acc:.4f}')
                        model.load_state_dict(best_model_wts)
                        return model, class_names # 返回模型和類別名稱
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    
    return model, class_names # 返回模型和類別名稱

if __name__ == '__main__':
    # ----- 1. 設定主要參數 -----
    config = {
        'data_dir': 'data/processed/food101', # 包含 train, val, (test) 的父目錄
        'train_subdir': 'train',
        'val_subdir': 'val',
        'num_classes': 100,  # 預期的食物類別數量
        'model_save_name': 'efficientnet_b0_food_v2_100_classes.pt', # 儲存模型的名稱
        'batch_size': 32,    # 降低 batch_size 以減少 MPS 上的記憶體壓力
        'num_epochs': 25,    # 可以根據需要調整
        'learning_rate': 5e-4, # 調整學習率
        'patience': 5,       # 早停的耐心值
        'label_smoothing': 0.1,
        'weight_decay': 1e-4
    }

    # ----- 執行訓練 -----
    print("Starting training process...")
    trained_model, class_names_list = train_efficientnet_model(
        data_dir=config['data_dir'], 
        train_subdir=config['train_subdir'], 
        val_subdir=config['val_subdir'], 
        num_classes=config['num_classes'], 
        model_save_name=config['model_save_name'],
        batch_size=config['batch_size'],
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        patience=config['patience'],
        label_smoothing=config['label_smoothing'],
        weight_decay=config['weight_decay']
    )
    
    if trained_model:
        print(f"Training finished. Best model saved to models/{config['model_save_name']}")
        # ----- (可選) 儲存類別名稱 -----
        try:
            classes_file_path = os.path.join('models', config['model_save_name'] + '.classes.txt')
            with open(classes_file_path, 'w') as f:
                for item in class_names_list:
                    f.write("%s\n" % item)
            print(f"Class names saved to {classes_file_path}")
        except Exception as e:
            print(f"Error saving class names: {e}")
    else:
        print("Training did not complete successfully or was stopped early without a model being returned.") 