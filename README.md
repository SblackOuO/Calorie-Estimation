# Food-Image-Calorie-Estimation-Personalized-Meal-Recommendation

This project uses image recognition to estimate food calories from user-uploaded images and generate personalized meal suggestions based on cultural preferences and ingredient history.

##  Features

-  Food type classification and object detection (YOLOv5 or Faster R-CNN)
-  Calorie estimation based on food type, portion size, and cooking method
-  Meal recommendation by cuisine (e.g., Taiwanese, Japanese, Western)
-  Optional ingredient-based refinement (via segmentation or user input)
-  Evaluation on accuracy, calorie error, and usability

##  Dataset

We use a combination of:
- [Food-101](https://data.vision.ee.ethz.ch/cvl/food-101/)
- [Food-101 Nutritional Information](https://www.kaggle.com/datasets/sanadalali/food-101-nutritional-information?fbclid=IwY2xjawKli6JleHRuA2FlbQIxMABicmlkETFiVER4YTRBbGpmUVYwTlRuAR58CAXne02G-MrhdALIj5w26y7h-rg54mgIY9Xm2I8ZzZTgs2cMpnSDzzungQ_aem_mr8c5qAUe476Udd-pBCxfA)

##  Models

- Mobilenetv2
- EfficientNet / ResNet for classification
- YOLOv5 / Faster R-CNN for food detection
