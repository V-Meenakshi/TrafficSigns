# 🚦 Object Detection (Traffic and Speed Sign Boards)

This project focuses on detecting **traffic signs** and **speed signboards** using a custom-trained **YOLOv8 model**. The system was developed to enhance road safety applications by enabling accurate recognition of traffic signals.

---

## 📌 Features

* ✅ Custom-trained **YOLOv8** model for traffic and speed sign detection
* ✅ Dataset curation and augmentation using **Roboflow**
* ✅ Model fine-tuning for **high precision and recall**
* ✅ Evaluation using **Precision, Recall, and mAP** metrics
* ✅ Deployment in **MATLAB simulation** for real-world testing

---

## 🛠️ Technologies Used

* **Python** – Model training and evaluation
* **YOLOv8 (Ultralytics)** – Object detection framework
* **Roboflow** – Dataset curation and augmentation
* **MATLAB** – Simulation and deployment

---

## 📂 Dataset

* Traffic sign and speed signboard images were collected, annotated, and augmented using **Roboflow**.
* Dataset split:

  * **70% Training**
  * **20% Validation**
  * **10% Testing**
* 📥 **Download Dataset:** [Click Here](https://drive.google.com/drive/folders/1V5my2jpX8KQSfR-KX4WWzG0Up3pgPdy_?usp=sharing)

---

## 📊 Model Performance

The trained model was evaluated using standard object detection metrics:

* **Precision**: High accuracy in detecting relevant objects
* **Recall**: Strong ability to detect most instances of traffic signs
* **mAP (Mean Average Precision)**: Reliable detection across multiple classes

---

## 🚀 How to Run

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/traffic-sign-detection.git
cd traffic-sign-detection
```

### 2️⃣ Install Dependencies

```bash
pip install ultralytics roboflow opencv-python matplotlib
```

### 3️⃣ Training the Model

```bash
yolo detect train data=dataset.yaml model=yolov8n.pt epochs=50 imgsz=640
```

### 4️⃣ Testing / Evaluation

```bash
yolo detect val model=runs/detect/train/weights/best.pt data=dataset.yaml
```

### 5️⃣ Inference on New Images

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=path/to/image.jpg
```

---

## 📺 MATLAB Simulation

* The trained YOLOv8 model was integrated with MATLAB for **simulation-based testing**.
* This allowed performance evaluation under **realistic traffic scenarios**.

---

## 📌 Applications

* 🚗 **Driver Assistance Systems (ADAS)**
* 🚦 **Smart Traffic Monitoring**
* 🛣️ **Autonomous Vehicles**

---

## 📖 References

* [YOLOv8 - Ultralytics Documentation](https://docs.ultralytics.com/)
* [Roboflow](https://roboflow.com/)
* [MATLAB Deep Learning Toolbox](https://www.mathworks.com/products/deep-learning.html)

---


## 📧 Contact

**Author:** Meenakshi Vinjamuri
If you find this project helpful, please ⭐ the repo and connect with me!

---

## 📜 License

Copyright © 2025 **Meenakshi Vinjamuri**

---

✨ *This project demonstrates the potential of deep learning in real-time traffic sign detection and its integration into intelligent transport systems.*

---

👉 Do you want me to insert your **Roboflow dataset link** directly into the README, or should I just leave it as a placeholder (`YOUR_DATASET_LINK`) for now?
