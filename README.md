Small educational program that gives you simple gui interface for
YOLO11 object detection model training and prediction.

Basic usage for model training:
1. Select weights file and check 'use weights?' if you want to 
train pretrained model.
2. Select training settings.
3. Click 'Train' button. If you need to abort training click 'Abort' button.

Basic usage for making predictions:
1. Load weights file and set 'use weights?' checked.
2. Click 'Select file' and select image for prediction.
3. Set needed confidence threshold.
4. Click 'Analyse' button to make prediction.
New window with image and boxes should appear.

'trained models' folder contains weights which can be used.
'cars examples' contains some cars photo for prediction examples.

Install from github:
pip install git+https://github.com/ilya205/yolo11-detection-simple-project-example
