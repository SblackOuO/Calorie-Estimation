import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
from torchvision import transforms, models
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
from tqdm import tqdm
import pandas as pd
import time

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']  # 支援中文顯示
plt.rcParams['axes.unicode_minus'] = False

class ModelComparator:
    def __init__(self, test_dir, train_dir):
        self.test_dir = test_dir
        self.train_dir = train_dir
        self.device = torch.device('mps' if torch.backends.mps.is_available() 
                                  else 'cuda' if torch.cuda.is_available() 
                                  else 'cpu')
        print(f"使用裝置: {self.device}")
        
        # 載入Food-101類別名稱
        classes_file = os.path.join('data', 'meta', 'classes.txt')
        with open(classes_file, 'r') as f:
            all_lines = [line.strip() for line in f.readlines() if line.strip()]
        
        self.full_class_names = all_lines
        self.num_full_classes = len(self.full_class_names)
        print(f"偵測到 {self.num_full_classes} 個完整類別名稱 (來自 classes.txt)")

        self.comparison_class_names = self.full_class_names[:100]
        self.num_comparison_classes = len(self.comparison_class_names)
        print(f"將使用 {self.num_comparison_classes} 個類別進行模型比較。")

        # 建立完整類別名稱到索引的映射 (用於 y_true)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.full_class_names)}
        
        # 預處理
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.models = {}
        self.results = {}
        
    def load_mobilenet_model(self, model_path):
        """載入MobileNetV2模型 (預期100個類別)"""
        print("載入MobileNetV2模型...")
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, self.num_comparison_classes) # 100個類別
        )
        
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()
        
        self.models['MobileNetV2'] = model
        print("MobileNetV2模型載入完成 (100個類別)")
        
    def load_efficientnet_model(self, model_path):
        """載入EfficientNet-B0模型 (預期101個類別)"""
        print("載入EfficientNet-B0模型...")
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        num_ftrs = model.classifier[1].in_features 
        # EfficientNet權重是101個類別
        model.classifier[1] = nn.Linear(num_ftrs, self.num_full_classes) # 101個類別 
        
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()
        
        self.models['EfficientNet-B0'] = model
        print("EfficientNet-B0模型載入完成 (101個類別)")
        
    def predict_single_image(self, model, image_path):
        """對單張圖片進行預測"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.val_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_idx = logits.argmax(dim=1).item()
            confidence = probabilities[0][predicted_idx].item()
            
        return predicted_idx, confidence, probabilities[0].cpu().numpy()
    
    def evaluate_model(self, model_name, model):
        """評估單個模型"""
        print(f"\n評估 {model_name}...")
        
        y_true = []
        y_pred = []
        y_proba = [] # For ROC later if needed, stores probabilities for all classes
        inference_times = []
        confidences = []
        misclassified = []
        
        # 只評估測試目錄中實際存在的類別 (這些類別名稱應存在於 self.full_class_names)
        test_classes_in_folder = [d for d in os.listdir(self.test_dir) 
                                  if os.path.isdir(os.path.join(self.test_dir, d)) and d in self.class_to_idx]
        
        print(f"測試目錄中找到 {len(test_classes_in_folder)} 個有效類別進行評估")
        
        for true_class_name in tqdm(test_classes_in_folder, desc=f'評估{model_name}'):
            # y_true 使用 full_class_names 的索引
            true_class_idx = self.class_to_idx[true_class_name]
            class_dir = os.path.join(self.test_dir, true_class_name)
                
            images = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for image_name in images:
                image_path = os.path.join(class_dir, image_name)
                
                start_time = time.time()
                # pred_idx 是模型自身輸出空間的索引 (MNv2: 0-99, ENet: 0-100)
                pred_idx, confidence, proba_vector = self.predict_single_image(model, image_path)
                inference_time = time.time() - start_time
                
                y_true.append(true_class_idx)
                y_pred.append(pred_idx)
                y_proba.append(proba_vector) # Store full probability vector
                inference_times.append(inference_time)
                confidences.append(confidence)
                
                # 比較時，如果true_class_name 不在 comparison_class_names 中 (不太可能，因為test_classes_in_folder做了篩選)
                # 或者 pred_idx 對應的 self.full_class_names[pred_idx] 不是 true_class_name
                if pred_idx >= len(self.full_class_names) or self.full_class_names[pred_idx] != true_class_name:
                    misclassified.append({
                        'true_class': true_class_name,
                        'pred_class': self.full_class_names[pred_idx] if pred_idx < len(self.full_class_names) else '超出範圍',
                        'confidence': confidence,
                        'image_path': image_path
                    })
        
        # 計算指標時，基於 self.num_comparison_classes (100個類別)
        # sklearn метрики будут игнорировать метки в y_true/y_pred, которых нет в labels
        report_labels_indices = list(range(self.num_comparison_classes))

        accuracy = accuracy_score(y_true, y_pred, normalize=True)
        # For precision, recall, f1, ensure they are calculated over the 100 common classes
        # We need to filter y_true and y_pred for these scores if we want to be super strict
        # or rely on `labels` argument and `zero_division`.
        # Let's filter for clarity for P/R/F1:
        filtered_y_true_for_report = []
        filtered_y_pred_for_report = []
        for yt, yp in zip(y_true, y_pred):
            if yt < self.num_comparison_classes: # Only consider true labels within the 100 classes
                filtered_y_true_for_report.append(yt)
                # If prediction is outside 100 classes, map it to an 'other' category or handle as misclassification
                # For simplicity here, we'll keep the original prediction for P/R/F1 calculation with labels arg.
                filtered_y_pred_for_report.append(yp if yp < self.num_comparison_classes else -1) # -1 for 'other' essentially

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=report_labels_indices, average='weighted', zero_division=0
        )
        # Accuracy is overall, so it's fine with original y_true, y_pred as it counts correct / total.
        # The accuracy_score will be slightly different if calculated on filtered lists vs full lists if ENet predicts class 101.
        # For a fair comparison against a 100-class system, accuracy should be calculated on predictions within these 100 classes.
        # Let's recalculate accuracy based on the common 100 classes scope for the report
        correct_predictions_in_100_scope = 0
        total_predictions_in_100_scope = 0
        for yt, yp in zip(y_true, y_pred):
            if yt < self.num_comparison_classes: # If the true label is one of the 100
                total_predictions_in_100_scope += 1
                if yt == yp:
                    correct_predictions_in_100_scope +=1
        
        reported_accuracy = correct_predictions_in_100_scope / total_predictions_in_100_scope if total_predictions_in_100_scope > 0 else 0

        avg_inference_time = np.mean(inference_times)
        avg_confidence = np.mean(confidences)
        
        self.results[model_name] = {
            'y_true': y_true, # Full list
            'y_pred': y_pred, # Full list
            'y_proba': np.array(y_proba),
            'accuracy': reported_accuracy, # Accuracy based on 100 classes
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_inference_time': avg_inference_time,
            'avg_confidence': avg_confidence,
            'misclassified': misclassified,
            'inference_times': inference_times,
            'confidences': confidences,
            'test_classes_evaluated': test_classes_in_folder
        }
        
        print(f"{model_name} 評估完成:")
        print(f"  報告準確率 (100類共同評估): {reported_accuracy:.4f}")
        print(f"  精確度 (100類加權): {precision:.4f}")
        print(f"  召回率 (100類加權): {recall:.4f}")
        print(f"  F1分數 (100類加權): {f1:.4f}")
        print(f"  平均推理時間: {avg_inference_time:.4f}秒")
        print(f"  平均信心度: {avg_confidence:.4f}")
        
    def plot_comparison_metrics(self):
        """繪製模型比較指標"""
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        axes = [ax1, ax2, ax3, ax4]
        
        for idx, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in models]
            bars = axes[idx].bar(models, values, color=['#FF6B6B', '#4ECDC4'])
            axes[idx].set_title(f'{metric.capitalize()}比較', fontsize=14, fontweight='bold')
            axes[idx].set_ylabel(metric.capitalize())
            axes[idx].set_ylim(0, 1)
            
            # 在柱狀圖上顯示數值
            for bar, value in zip(bars, values):
                axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                              f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.test_dir}/../model_comparison_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_confusion_matrices(self):
        """繪製混淆矩陣 (基於100個比較類別)"""
        fig, axes = plt.subplots(1, len(self.results), figsize=(10 * len(self.results) + 2, 8))
        if len(self.results) == 1:
            axes = [axes] # Ensure axes is always iterable
        
        report_labels_indices = list(range(self.num_comparison_classes))

        for idx, (model_name, result) in enumerate(self.results.items()):
            ax = axes[idx]
            # Generate CM for the 100 comparison classes
            cm = confusion_matrix(result['y_true'], result['y_pred'], labels=report_labels_indices)
            
            # Normalize the confusion matrix
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_normalized = np.zeros_like(cm, dtype=float)
            # Avoid division by zero for rows with no samples
            valid_rows = row_sums.flatten() > 0
            if np.any(valid_rows):
                 cm_normalized[valid_rows, :] = cm[valid_rows, :] / row_sums[valid_rows, :]
            
            top_classes_to_plot = min(20, self.num_comparison_classes)
            cm_plot = cm_normalized[:top_classes_to_plot, :top_classes_to_plot]
            class_labels_plot = self.comparison_class_names[:top_classes_to_plot]
            
            sns.heatmap(cm_plot, 
                       xticklabels=class_labels_plot,
                       yticklabels=class_labels_plot,
                       annot=True, 
                       fmt='.2f', # Format annotations to 2 decimal places
                       cmap='Blues',
                       ax=ax,
                       cbar=True, # Show color bar
                       square=True, # Make cells square
                       linewidths=.5 # Add lines between cells
            )
            
            ax.set_title(f'{model_name} 混淆矩陣 ({self.num_comparison_classes}類基礎，顯示前{top_classes_to_plot})', fontsize=14)
            ax.set_xlabel('預測類別', fontsize=12)
            ax.set_ylabel('實際類別', fontsize=12)
            
            ax.tick_params(axis='x', rotation=45, ha='right')
            ax.tick_params(axis='y', rotation=0)
        
        plt.tight_layout(pad=3.0) # Add padding
        plt.savefig(f'{self.test_dir}/../confusion_matrices.png', dpi=300, bbox_inches='tight')
        print(f"混淆矩陣已儲存至: {self.test_dir}/../confusion_matrices.png")
        plt.show()
        
    def plot_inference_time_comparison(self):
        """繪製推理時間比較"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 平均推理時間比較
        models = list(self.results.keys())
        avg_times = [self.results[model]['avg_inference_time'] for model in models]
        
        bars = ax1.bar(models, avg_times, color=['#FF6B6B', '#4ECDC4'])
        ax1.set_title('平均推理時間比較', fontsize=14, fontweight='bold')
        ax1.set_ylabel('時間 (秒)')
        
        for bar, time in zip(bars, avg_times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                    f'{time:.4f}s', ha='center', va='bottom', fontweight='bold')
        
        # 推理時間分布
        for model_name in models:
            times = self.results[model_name]['inference_times']
            ax2.hist(times, alpha=0.7, label=model_name, bins=50)
        
        ax2.set_title('推理時間分布', fontsize=14, fontweight='bold')
        ax2.set_xlabel('推理時間 (秒)')
        ax2.set_ylabel('頻率')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.test_dir}/../inference_time_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_confidence_comparison(self):
        """繪製信心度比較"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 平均信心度比較
        models = list(self.results.keys())
        avg_confidences = [self.results[model]['avg_confidence'] for model in models]
        
        bars = ax1.bar(models, avg_confidences, color=['#FF6B6B', '#4ECDC4'])
        ax1.set_title('平均預測信心度比較', fontsize=14, fontweight='bold')
        ax1.set_ylabel('信心度')
        ax1.set_ylim(0, 1)
        
        for bar, conf in zip(bars, avg_confidences):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{conf:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 信心度分布
        for model_name in models:
            confidences = self.results[model_name]['confidences']
            ax2.hist(confidences, alpha=0.7, label=model_name, bins=50)
        
        ax2.set_title('預測信心度分布', fontsize=14, fontweight='bold')
        ax2.set_xlabel('信心度')
        ax2.set_ylabel('頻率')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.test_dir}/../confidence_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def analyze_misclassifications(self):
        """分析錯誤分類"""
        print("\n=== 錯誤分類分析 ===")
        
        for model_name, result in self.results.items():
            print(f"\n{model_name}:")
            misclassified = result['misclassified']
            print(f"錯誤分類數量: {len(misclassified)}")
            
            # 統計最常見的錯誤分類對
            error_pairs = Counter([(m['true_class'], m['pred_class']) for m in misclassified])
            print("最常見的錯誤分類對 (前10個):")
            for (true_class, pred_class), count in error_pairs.most_common(10):
                print(f"  {true_class} -> {pred_class}: {count}次")
                
    def generate_detailed_report(self):
        """生成詳細報告 (基於100個比較類別)"""
        print("\n=== 詳細分類報告 ===")
        
        report_labels_indices = list(range(self.num_comparison_classes))
        report_target_names = self.comparison_class_names

        for model_name, result in self.results.items():
            print(f"\n{model_name}:")
            print("-" * 50)
            
            # Generate classification report for the 100 comparison classes
            report_str = classification_report(
                result['y_true'], 
                result['y_pred'], 
                labels=report_labels_indices,
                target_names=report_target_names,
                zero_division=0
            )
            print(report_str)
            
    def save_results_to_csv(self):
        """將結果儲存為CSV"""
        summary_data = []
        
        for model_name, result in self.results.items():
            summary_data.append({
                '模型': model_name,
                '準確率': result['accuracy'],
                '精確度': result['precision'],
                '召回率': result['recall'],
                'F1分數': result['f1'],
                '平均推理時間(秒)': result['avg_inference_time'],
                '平均信心度': result['avg_confidence'],
                '錯誤分類數量': len(result['misclassified'])
            })
            
        df = pd.DataFrame(summary_data)
        csv_path = f'{self.test_dir}/../model_comparison_results.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n結果已儲存至: {csv_path}")
        
        return df
        
    def run_comparison(self, mobilenet_path, efficientnet_path):
        """執行完整的模型比較"""
        print("開始模型比較...")
        
        # 載入模型
        self.load_mobilenet_model(mobilenet_path)
        self.load_efficientnet_model(efficientnet_path)
        
        # 評估模型
        for model_name, model in self.models.items():
            self.evaluate_model(model_name, model)
            
        # 生成報告和圖表
        print("\n生成比較圖表...")
        self.plot_comparison_metrics()
        self.plot_confusion_matrices()
        self.plot_inference_time_comparison()
        self.plot_confidence_comparison()
        
        # 分析結果
        self.analyze_misclassifications()
        self.generate_detailed_report()
        
        # 儲存結果
        summary_df = self.save_results_to_csv()
        
        print("\n=== 模型比較摘要 ===")
        print(summary_df.to_string(index=False))
        
        return self.results

if __name__ == "__main__":
    # 設定路徑
    TEST_DIR = os.path.join('data', 'processed', 'food101', 'test')
    TRAIN_DIR = os.path.join('data', 'processed', 'food101', 'train')
    MOBILENET_PATH = os.path.join('models', 'food_classifier.pt')
    EFFICIENTNET_PATH = os.path.join('models', 'efficientnet_b0_food_v2_100_classes.pt')
    
    # 檢查檔案存在
    if not os.path.exists(MOBILENET_PATH):
        print(f"錯誤: MobileNetV2模型檔案不存在: {MOBILENET_PATH}")
        exit(1)
        
    if not os.path.exists(EFFICIENTNET_PATH):
        print(f"錯誤: EfficientNet模型檔案不存在: {EFFICIENTNET_PATH}")
        exit(1)
        
    if not os.path.exists(TEST_DIR):
        print(f"錯誤: 測試資料目錄不存在: {TEST_DIR}")
        exit(1)
    
    # 執行比較
    comparator = ModelComparator(TEST_DIR, TRAIN_DIR)
    results = comparator.run_comparison(MOBILENET_PATH, EFFICIENTNET_PATH)
    
    print("\n模型比較完成！")
    print("生成的檔案:")
    print("- model_comparison_metrics.png: 指標比較圖")
    print("- confusion_matrices.png: 混淆矩陣")
    print("- inference_time_comparison.png: 推理時間比較")
    print("- confidence_comparison.png: 信心度比較")
    print("- model_comparison_results.csv: 詳細結果數據") 