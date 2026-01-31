# Real-Time Face Recognition (Edge ML)

CPU-optimized face recognition pipeline for edge deployment on Raspberry Pi 3B+. Achieves real-time UX through frame skipping with <50ms inference latency on processed frames.

## Problem & Constraints

**Deploy face recognition on:**
- Raspberry Pi 3B+ (1 GB RAM, CPU-only, no GPU)
- Target: Real-time operation (<50ms per frame)
- Handle partial occlusions (masks, glasses)

## Technical Approach

### Why Haar Cascade + KNN (not CNNs)?

| Factor | KNN (Selected) | ResNet-based CNN |
|--------|----------------|------------------|
| Model Size | <1 MB | ~90 MB |
| CPU Inference | 40 ms | 300 ms |
| Training Data | 100 samples/person | 1000+ samples/person |
| Memory Usage | ~20 MB | ~500 MB |

**Decision:** Edge constraints (CPU-only, limited RAM) made classical ML the only viable choice.

### Pipeline

```
Video (30 FPS) → Haar Cascade (20ms) → Preprocess (5ms) → KNN (15ms) → Output
                                                             ↓
                                    Frame skip: 15 FPS effective processing
```

**Key optimizations:**
- Grayscale conversion (3× faster than RGB)
- Frame skipping (maintains real-time UX)
- Vectorized KNN inference (batch predictions)
- Cascade tuning (`minNeighbors=5` reduces false positives)

## Performance

### Accuracy
- Full face, frontal: **95%**
- Partial occlusion (mask): **75%**
- Glasses: **90%**
- Side angle (±30°): **70%**

### Latency (Raspberry Pi 3B+)
- Detection (Haar): 20 ms
- Preprocessing: 5 ms
- KNN inference: 15 ms
- **Total: 40 ms** (per processed frame)

### Failure Modes Identified
1. **Low lighting**: 40% accuracy drop (Haar cascade relies on intensity gradients)
2. **Side angles >30°**: Detection rate 20% (cascade trained on frontal faces)
3. **Multiple faces (>3)**: Linear scaling breaks real-time target

## Implementation

### Setup
```bash
pip install -r requirements.txt
python add_faces.py  # Collect 100 samples/person
python test.py       # Run inference (press 'o' to log, 'q' to quit)
```

### Data Collection (`add_faces.py`)
- Captures 100 face samples (50×50 grayscale) per person
- Samples every 10th frame to ensure diversity
- Stores as pickled numpy arrays

### Inference (`test.py`)
- Loads training data, trains KNN (k=5, distance-weighted)
- Real-time recognition with confidence scoring
- Logs attendance to CSV with timestamps

## Design Trade-offs

| Decision | Why | What Was Sacrificed |
|----------|-----|---------------------|
| Haar Cascade | 20ms speed | Angle robustness |
| Raw pixels (2500D) | Simple pipeline | Compact representation |
| KNN | No training overhead | Scalability (O(n) search) |
| 50×50 resolution | Memory + speed | Fine detail |
| Frame skipping | Real-time UX | Temporal smoothness |

## Key Learnings

**Why not deep learning?**
Prototyped MobileNetV2 + SVM (90 MB model) → 300ms inference on CPU. Would need GPU (Jetson Nano) or model quantization for real-time performance.

**Scalability limitations:**
KNN has O(n) search complexity. For 1000+ identities, would need:
- Approximate Nearest Neighbors (FAISS) for O(log n) search
- Face embeddings (FaceNet 128D) instead of 2500D raw pixels
- Hardware upgrade to Jetson Nano for CNN deployment

**Robustness insights:**
- 100 samples/person sufficient (learning curve plateaus)
- Data augmentation (rotation, brightness) showed <2% gain
- Haar cascade misses 80% of side-profile faces

## Future Improvements

**Short-term (no hardware change):**
1. MTCNN detection (better angle invariance)
2. Confidence thresholding (reject predictions <80%)
3. HOG features instead of raw pixels (15% accuracy gain in tests)

**Long-term (hardware upgrade):**
1. Deploy on Jetson Nano → FaceNet embeddings (10× speedup)
2. FAISS indexing for large-scale deployment
3. Anti-spoofing (liveness detection)

## Reproducibility

**Environment:**
- Raspberry Pi 3B+ (Quad-core ARM Cortex-A53, 1 GB RAM)
- Python 3.8, OpenCV 4.5 (CPU build)

**Known Issues:**
- Haar cascade path varies across OpenCV installations
- Pickle protocol compatibility (use protocol=4)
- Low FPS if video resolution >640×480

## References

**Core Techniques:**
- [Viola-Jones Face Detection (2001)](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf) – Haar Cascade
- [KNN for Face Recognition](https://ieeexplore.ieee.org/document/7029427) – Performance analysis

**Edge ML Resources:**
- [OpenCV Face Recognition Guide](https://docs.opencv.org/4.x/da/d60/tutorial_face_main.html)
- [TensorFlow Lite on Raspberry Pi](https://www.tensorflow.org/lite/guide/python)

---

**Author:** Saksham Bajaj | [LinkedIn](https://linkedin.com/in/saksham-bjj) | [GitHub](https://github.com/SakshamBjj)