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
- [UECFood256](http://foodcam.mobi/dataset256.html)
- [ChineseFoodNet](https://github.com/xtudbx/ChineseFoodNet)
- Custom daily meal photos

##  Models

- MobileNetV2 / ResNet for classification
- YOLOv5 / Faster R-CNN for food detection
- Calorie lookup table for estimation
- Optional: Mask R-CNN for ingredient segmentation

##  Setup

```bash
git clone https://github.com/<your-team-id>/food-calorie-ai.git
cd food-calorie-ai
pip install -r requirements.txt
