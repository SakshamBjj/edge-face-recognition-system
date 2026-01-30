# System Architecture

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────┐
│  Input: Video Frame (640×480 RGB, 30 FPS)              │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Grayscale Conversion (5ms)                             │
│  • RGB → Gray (3× faster downstream processing)         │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Face Detection: Haar Cascade (20ms)                    │
│  • scaleFactor=1.3, minNeighbors=5                      │
│  • Output: [(x, y, w, h), ...] bounding boxes           │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Per-Face Processing (5ms)                              │
│  • Crop face region                                     │
│  • Resize to 50×50                                      │
│  • Flatten to 2500D vector                              │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  KNN Classification (15ms)                              │
│  • k=5, weights='distance'                              │
│  • Distance-based confidence scoring                    │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Output: {name, confidence, bbox}                       │
│  Total Latency: 40ms per processed frame                │
└─────────────────────────────────────────────────────────┘
```

## Design Decisions

### 1. Face Detection: Haar Cascade

**Selected over:**
- MTCNN (50ms, better angle robustness)
- YOLO (100ms CPU, excellent but too slow)

**Configuration:**
```python
scaleFactor=1.3  # Balance: detection rate vs speed
minNeighbors=5   # Reduce false positives (higher = stricter)
```

**Trade-off:** 20ms speed vs poor side-angle detection (70% at ±30°)

---

### 2. Feature Representation: Raw Pixels

**Approach:** 50×50 grayscale → 2500D vector (no feature extraction)

**Tested alternatives:**
- HOG features: 15% accuracy gain, 10ms overhead → not worth complexity
- FaceNet embeddings: 128D (compact) but requires CNN (300ms CPU)

**Decision:** Raw pixels sufficient for prototype with 100 samples/person

---

### 3. Classifier: KNN

**Hyperparameter tuning:**
- k=1: Predictions flickered (sensitive to outliers)
- k=3: Better stability, occasional errors
- **k=5: Optimal** (validated via confusion matrix)
- k=7: Over-smoothing, 3% accuracy drop

**Distance weighting:** `weights='distance'` gives closer neighbors more influence

**Scalability limit:** O(n) search → 5 people (15ms), 1000 people (300ms, too slow)

---

### 4. Frame Skipping Strategy

**Problem:** 40ms processing > 33ms frame interval (30 FPS input) → can't keep up

**Solution:** Process every 2nd frame
- Input: 30 FPS
- Processing: 15 FPS effective (66ms interval available)
- User experience: Still feels real-time

**Trade-off:** Temporal smoothness vs meeting latency target

---

## Failure Mode Analysis

### 1. Low Lighting
**Symptoms:** Accuracy drops 95% → 55%  
**Root cause:** Haar cascade relies on intensity gradients (fails in uniform dark)  
**Mitigation:** IR camera, CLAHE histogram equalization, or lower detection threshold

### 2. Side Angles (>30°)
**Symptoms:** 20% detection rate, 70% accuracy when detected  
**Root cause:** Haar cascade trained on frontal faces only  
**Mitigation:** Multi-angle cascades, MTCNN (keypoint-based), or 3D pose estimation

### 3. Partial Occlusions (Masks)
**Symptoms:** 95% → 75% accuracy  
**Root cause:** KNN pixel similarity → missing lower face breaks pattern  
**Mitigation:** Train on masked faces (80%), upper-face-only cascade, or attention mechanisms

### 4. Multiple Faces (>3)
**Symptoms:** 3 faces (45ms), 5 faces (75ms) → breaks real-time  
**Root cause:** Linear KNN scaling per face  
**Mitigation:** Batch predictions, prioritize largest face, or upgrade to FAISS

---

## Optimization Breakdown

| Stage | Baseline | Optimized | Technique |
|-------|----------|-----------|-----------|
| Grayscale | 15ms | 5ms | Vectorized cv2.cvtColor |
| Detection | 40ms | 20ms | Tuned minNeighbors=5 |
| Preprocessing | 10ms | 5ms | Cached resize |
| KNN | 25ms | 15ms | Vectorized sklearn |
| **Total** | **90ms** | **45ms** | (40ms with final tuning) |

**Frame skipping:** Achieves real-time UX despite 40ms > 33ms constraint

---

## Scalability Analysis

### Current System (5-10 people) ✅
- 100 samples × 5 people = 500 vectors → 15ms KNN search
- Works on Raspberry Pi 3B+

### Medium Scale (50-100 people) ⚠️
- 5,000 vectors → 75ms (too slow)
- **Solution:** FAISS with 128D embeddings (FaceNet) → O(log n)
- **Hardware:** Jetson Nano (GPU for FaceNet)

### Large Scale (1000+ people) ❌
- Requires distributed system + cloud backend
- Current KNN approach infeasible

---

## Production Upgrade Path

**Phase 1: Better Detection**
- Replace Haar with MTCNN → angle robustness

**Phase 2: Compact Features**
- Switch to HOG or FaceNet embeddings → 20× smaller

**Phase 3: Fast Search**
- Deploy FAISS indexing → O(log n) vs O(n)

**Phase 4: Hardware**
- Jetson Nano (GPU) → 30 FPS without frame skipping

---

## Key Metrics (Interview Ready)

**Performance:**
- Latency: 40ms (meets <50ms target)
- Effective FPS: 15 (frame skipping)
- Model size: <1 MB

**Accuracy:**
- Full face: 95%
- Mask: 75%
- Glasses: 90%
- Side angle: 70%

**Robustness:**
- Low light: 55% (-40%)
- False rejection: 8%
- Cross-person error: <2%

---

## References

**Core Papers:**
- Viola-Jones (2001) – Haar Cascade foundation
- FaceNet (2015) – Deep learning embeddings
- MTCNN (2016) – Robust detection

**Implementation:**
- [scikit-learn KNN](https://scikit-learn.org/stable/modules/neighbors.html)
- [OpenCV Face Detection](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
