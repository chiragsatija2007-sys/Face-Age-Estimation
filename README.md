# 🔍 Face Age Detection System

Real-time face and age detection using deep learning. Production-ready edge AI application with 5 major software engineering innovations.

**Status:** ✅ Complete | Optimized for 30+ FPS | Offline-First

---

## ✨ Features
- ⚡ Real-time face detection with OpenCV DNN
- 🧠 Age estimation using GoogLeNet CNN (ONNX)
- 🚀 30+ FPS optimized performance
- 🎨 Custom cyberpunk HUD UI design
- 🔒 Offline architecture (no cloud, no data transmission)
- 📊 Confidence scoring with intelligent thresholding

---

## 🏆 5 Major Software Engineering Innovations

### 1. FPS Caching System (Frame-Skipping)
**Problem:** Inference bottleneck = 1-2 FPS  
**Solution:** Process every 3rd frame, cache results in between  
**Result:** ✅ 30+ FPS performance boost

### 2. UI Smoothing with Deque
**Problem:** Confidence scores flicker, unstable age labels  
**Solution:** Deque (maxlen=8) → median/mode filtering  
**Result:** ✅ Flicker-free, stable output

### 3. Tensor Pre-Processing (BGR→RGB Conversion)
**Problem:** Color space mismatch broke face recognition  
**Solution:** swapRB=True in cv2.dnn.blobFromImage  
**Result:** ✅ Correct color matrix conversion

### 4. Softmax De-duplication & Intelligent Thresholding
**Problem:** Confidence scores artificially low (~18%)  
**Solution:** Removed redundant Softmax, lowered threshold to 35%  
**Result:** ✅ Realistic edge-computing variance handling

### 5. Cyberpunk Dragonfruit HUD Design
**Problem:** Standard OpenCV text invisible on light backgrounds  
**Solution:** Custom corner-bracket design + dynamic background boxes  
**Result:** ✅ 100% legible UI in any lighting condition

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.x |
| **Face Detection** | OpenCV DNN |
| **Age Model** | GoogLeNet CNN (ONNX) |
| **UI Framework** | Tkinter |
| **Libraries** | NumPy, OpenCV |
| **Performance** | 30+ FPS, ~50ms inference |

---

## 📦 Installation

### Requirements
- Python 3.7+
- 200MB free RAM

### Setup

```bash
# Clone or download the repository

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Option 1: Real-Time Webcam Detection
```bash
python main.py
```
**Controls:**
- Detects faces in real-time
- Shows age prediction with confidence score
- Press **Q** to quit

### Option 2: Test with Static Images
```bash
python select_and_test.py
```
**How it works:**
1. Window opens with file browser
2. Select an image from `test/` folder
3. System detects faces and estimates ages
4. Shows results with confidence scores

---

## 📁 Folder Structure
