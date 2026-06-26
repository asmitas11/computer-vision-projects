# 👁️ Computer Vision Suite — Face Detection, Recognition & Edge Detection

A hands-on exploration of **classical and deep learning computer vision techniques** implemented in Python and Google Colab. This project covers 6 face detection algorithms, face recognition using embeddings, and 5 edge detection methods — each tested on real images with annotated outputs.

---

## What's Inside

| File / Notebook | What it covers |
|---|---|
| `face_detection.ipynb` | Haar, HOG, SSD, MTCNN, MMOD, DeepFace — all 6 detectors on the same image |
| `face_recognition.ipynb` | Encode known faces → identify people in a group photo |
| `edge_detection.ipynb` | Sobel, Prewitt, Roberts, Canny, LoG — side-by-side comparison |
| `realtime_detection.ipynb` | Live webcam detection (Colab JS + local) |
| `models/` | Model files: `haarcascade_frontalface_default.xml`, `caffe.prototxt`, `res10_300x300_ssd_iter_140000.caffemodel`, `mmod_human_face_detector.dat` |

---

## Face Detection — 6 Algorithms Compared

All 6 detectors were run on the same group photo (`All.jpg`) to make comparison fair.

### 1. Haar Cascade
The oldest method — uses handcrafted Haar features (edges, lines, rectangles) in a sliding window cascade trained with AdaBoost. Ships with OpenCV, no download needed.

- ✅ Extremely fast, works on CPU
- ❌ Many false positives, fails on non-frontal faces and poor lighting
- Tunable via `scaleFactor` and `minNeighbors`

### 2. HOG + SVM (dlib)
Histogram of Oriented Gradients — computes gradient direction histograms in local image patches, feeds them into an SVM. More robust than Haar because gradient orientations are lighting-invariant.

- ✅ Fewer false positives than Haar, no model file needed
- ❌ Still struggles with profile faces (>30° rotation)
- `upsample=1` catches smaller faces at cost of speed

### 3. SSD — Single Shot Detector (OpenCV DNN + Caffe)
First deep learning detector in this project. A ResNet-10 backbone trained on a large face dataset — detects faces in one forward pass and returns a **confidence score** per detection.

- ✅ Best accuracy/speed balance among classical+DL methods
- ✅ Returns confidence scores — filterable by threshold
- ❌ Requires downloading `.prototxt` and `.caffemodel` files separately

### 4. MTCNN — Multi-task Cascaded CNN
Three-stage cascade: Proposal Net → Refine Net → Output Net. The only detector here that returns **facial landmarks** (eyes, nose, mouth corners).

- ✅ Accurate at multiple scales, handles small faces
- ✅ Returns 5 keypoints per face — useful for face alignment before recognition
- ❌ Slower than Haar/HOG

### 5. MMOD — CNN Face Detector (dlib)
dlib's deep CNN trained with Max-Margin Object Detection loss. Most accurate dlib method — handles profile faces and partial occlusion that HOG misses.

- ✅ Highest accuracy among all dlib methods
- ✅ Handles difficult angles and occlusion
- ❌ Slowest detector here without GPU, requires `.dat` model file (~68MB)

### 6. DeepFace
High-level wrapper supporting multiple backends (opencv, ssd, dlib, mtcnn, retinaface). Also runs **attribute analysis** — age, gender, and dominant emotion per face.

- ✅ Age / Gender / Emotion estimation in one call
- ✅ Flexible backend switching
- ❌ Heavier dependency, slower for pure detection

---

## Detector Comparison

| Detector | Speed | Accuracy | Confidence Score | Extra Output | Model File Needed |
|---|---|---|---|---|---|
| Haar Cascade | ⚡⚡⚡ | ⭐⭐ | ❌ | — | ❌ (built into OpenCV) |
| HOG (dlib) | ⚡⚡ | ⭐⭐⭐ | ❌ | — | ❌ |
| SSD (Caffe) | ⚡⚡ | ⭐⭐⭐⭐ | ✅ | — | ✅ `.caffemodel` |
| MTCNN | ⚡ | ⭐⭐⭐⭐ | ✅ | 5 facial landmarks | ❌ (auto-download) |
| MMOD (dlib) | 🐢 | ⭐⭐⭐⭐⭐ | ✅ | — | ✅ `.dat` |
| DeepFace | ⚡ | ⭐⭐⭐⭐ | ✅ | Age, Gender, Emotion | ❌ (auto-download) |

---

## Face Recognition

Using the `face_recognition` library (built on dlib's 128-d face embeddings).

**How it works:**
1. Load reference images of known people (`modi.jpg`, `trump.jpg`)
2. Compute a 128-dimensional embedding for each known face
3. For each face detected in the target image, compute its embedding
4. Match using Euclidean distance — if distance < 0.5, it's a match

**Tested on:** A group photo (`Modi-Trump.jpg`) containing both known faces.  
Output: Bounding boxes with name labels and confidence overlay.

> The `face_recognition` library also supports HOG and CNN detection modes — tested with both `upsample` levels to show the effect on small/distant faces.

---

## Edge Detection — 5 Algorithms

All 5 methods were applied to the same two grayscale images and displayed in a side-by-side comparison grid.

| Algorithm | Year | Kernel Size | Key Idea |
|---|---|---|---|
| Roberts Cross | 1963 | 2×2 | Diagonal gradient — oldest method, very noise-sensitive |
| Prewitt | 1970 | 3×3 | Uniform gradient kernels in X and Y |
| Sobel | 1968 | 3×3 | Weighted gradient — central row/column gets 2× weight |
| Laplacian of Gaussian (LoG) | 1980 | Variable | Gaussian blur first, then second derivative |
| Canny | 1986 | Variable | Gaussian + Non-max suppression + hysteresis — gold standard |

**Key observation from experiments:**
- Roberts and Prewitt are noisier on real images because their kernels are smaller and simpler
- Sobel is a strict improvement over Prewitt for most images
- LoG's Gaussian pre-blur makes it significantly less sensitive to noise than raw Laplacian
- Canny produces the cleanest single-pixel-wide edges because of its non-maximum suppression step — which is why it's still the default choice in most CV pipelines today

---

**Model files** — download and place in `models/`:

| File | Source |
|---|---|
| `haarcascade_frontalface_default.xml` | Included with OpenCV |
| `caffe.prototxt` + `res10_300x300_ssd_iter_140000.caffemodel` | [OpenCV GitHub](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector) |
| `mmod_human_face_detector.dat` | [dlib.net](http://dlib.net/files/mmod_human_face_detector.dat.bz2) |

MTCNN and DeepFace download weights automatically on first use.

---

## Key Learnings

- Haar features are hand-engineered — deep learning models learn better features automatically from data
- HOG is more robust to lighting than Haar because gradient orientations are largely illumination-invariant
- MTCNN's facial landmarks are what enable reliable face alignment before recognition — skipping alignment hurts recognition accuracy significantly
- Face recognition works via **embedding similarity**, not pixel comparison — two photos of the same person in different lighting will still have embeddings close together
- Canny is the standard edge detector not because it's the newest, but because non-maximum suppression gives clean, thin, single-pixel edges that other methods don't
- The right detector depends on your constraint: Haar if you need speed, MMOD if you need accuracy, SSD if you need a balance with confidence scores

---

## Tech Stack

`Python` · `OpenCV` · `dlib` · `MTCNN` · `DeepFace` · `face_recognition` · `scikit-image` · `NumPy` · `Matplotlib` · `Google Colab`

---

## License

MIT