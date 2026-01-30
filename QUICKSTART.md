# Quickstart

## Setup
```bash
pip install -r requirements.txt
```

## Collect Faces
```bash
python add_faces.py
# Enter name when prompted
# System collects 100 samples automatically (~10 seconds)
```

## Run Recognition
```bash
python test.py
# Press 'o' to log attendance
# Press 'q' to quit
```

## Troubleshooting

**"Haar cascade file not found"**
```python
# Check OpenCV path
import cv2
print(cv2.data.haarcascades)
```

**"Cannot access webcam"**
```bash
# Linux: Add user to video group
sudo usermod -a -G video $USER

# Verify camera
ls /dev/video*
```

**Low FPS on Raspberry Pi**
```python
# Reduce resolution in test.py
video.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
```

## Raspberry Pi Deployment

**Install dependencies:**
```bash
sudo apt-get update
sudo apt-get install python3-opencv python3-pip
pip3 install scikit-learn==1.0.2 numpy==1.21.5
```

**For Pi Camera Module:**
```bash
# Enable in raspi-config
sudo raspi-config
# Interfacing Options → Camera → Enable
```

---

**More details:** See [ARCHITECTURE.md](ARCHITECTURE.md) for design decisions and optimization breakdown.
