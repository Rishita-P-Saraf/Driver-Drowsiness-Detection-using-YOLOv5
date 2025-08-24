# Driver Drowsiness Detection using YOLOv5

This project implements a **Driver Drowsiness Detection System** using the **YOLOv5 object detection model**.  
The system can detect whether a driver is **awake** or **drowsy** in real-time using a webcam feed.

---

## 🚀 Features
- Real-time driver monitoring using YOLOv5.
- Custom dataset collection (awake vs drowsy states).
- Image annotation with **LabelImg**.
- Train YOLOv5 on custom datasets.
- Live webcam detection for drowsiness monitoring.

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
```
### 2. Install Dependencies
```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install -r requirements.txt
```

### 3. Install LabelImg for Annotation
```bash
git clone https://github.com/tzutalin/labelImg
pip install pyqt5 lxml --upgrade
cd labelImg && pyrcc5 -o libs/resources.py resources.qrc
```
---

## 📂 Project Structure
```bash
Driver-Drowsiness-Detection/
│── yolov5/                  # YOLOv5 repository
│── data/
│   ├── images/              # Collected images (awake/drowsy)
│   │   ├── train/           # Training images
│   │   ├── val/             # Validation images
│   ├── labels/              # YOLO format labels
│   │   ├── train/
│   │   ├── val/
│── dataset.yml              # Dataset configuration file
│── runs/train/exp*/         # Training outputs and weights
│── README.md                # Project documentation
```
---

## 🖼️ Dataset Collection

### 1. Capture Awake images:
```python
cap = cv2.VideoCapture(0)
# Loop to capture 20 images of "awake" state
```

### 2. Capture Drowsy images:
```python
cap = cv2.VideoCapture(0)
# Loop to capture 20 images of "drowsy" state
```

### 3. Annotate the images using LabelImg and save labels in YOLO format.

---

## 🏋️ Training the Model

Run YOLOv5 training:

```bash
cd yolov5
python train.py --img 320 --batch 16 --epochs 5 --data dataset.yml --weights yolov5s.pt --workers 2
```
Training results (metrics, loss curves, and weights) will be saved inside:
```bash
runs/train/exp*/
```
---

## 📊 Testing & Inference

### 1. Load Pre-trained Model
```python
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
results = model('your_image.jpg')
results.print()
results.show()
```

### 2. Run Real-Time Detection
```python
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    results = model(frame)
    cv2.imshow('YOLO', np.squeeze(results.render()))
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

### 3. Load Custom Trained Model
```python
model = torch.hub.load('yolov5', 
                       'custom', 
                       path='runs/train/exp2/weights/best.pt', 
                       source='local')

```
---

## ⚡ Future Improvements

- Improve dataset size for better accuracy.
- Add alert system (sound/vibration) when drowsiness is detected.
- Deploy the model on edge devices (Raspberry Pi, Jetson Nano).

---

## 📝 Author

👤 Rishita Priyadarshini Saraf

📧 rishitasarafp@gmail.com
---

## 📜 License

This project is licensed under the MIT License.
