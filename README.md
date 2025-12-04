---
title: stable-diffusion-image-generator
app_file: src/sdgen/main.py
sdk: gradio
sdk_version: 3.50.2
---
# 🎨 Stable Diffusion Image Generator

AI system built using **Stable Diffusion (HuggingFace Diffusers)** and a modern **Gradio UI**.
This project generates high-quality images from text prompts and includes advanced capabilities such as:

* Style presets
* Image-to-Image generation
* Super-resolution upscaling (RealESRGAN)
* Prompt history & metadata tracking
* Seed reproducibility
* LoRA extension support

---

# Feature Details

## 1️⃣ **Text-to-Image Generation**

* Supports prompts & negative prompts
* Adjustable steps, CFG scale, resolution
* Seed for reproducibility
* Preset selection panel

## 2️⃣ **Image-to-Image (Img2Img)**

Transform uploaded images using prompts, e.g.:

* “Make this photo look cyberpunk”
* “Convert this portrait into anime style”
* “Turn into oil painting style”

## 3️⃣ **Super-Resolution Upscaling**

Improve output quality significantly:

* 1.5×
* 2×
* 4×
  Powered by **RealESRGAN**.

## 4️⃣ **Style Presets**

One-click artistic styles:

* Anime
* Realistic photography
* Pixar / 3D
* Oil painting
* Cyberpunk neon

## 5️⃣ **Prompt History & Metadata Tracking**

Every generation stores:

* Prompt
* Negative prompt
* Configuration
* Seed
* Generated image

## 6️⃣ **LoRA Support**

Load and use custom LoRA fine-tuned models:

* Styles
* Artists
* Characters
* Themes

---

# 🧩 Project Architecture

```
stable-diffusion-image-generator/
│
├── app/
│   ├── core/
│   │   └── __init__.py
│   │
│   ├── pipeline.py
│   │   # Loads & initializes Stable Diffusion (FP16, GPU, model configs)
│   │
│   ├── generator.py
│   │   # Text-to-image inference logic
│   │
│   ├── img2img.py
│   │   # Image-to-image transformation logic
│   │
│   ├── ui.py
│   │   # Complete Gradio interface with multiple tabs:
│   │   # Text2Img, Img2Img, Upscaling, History, About
│   │
│   ├── presets/
│   │   ├── styles.py
│   │       # Predefined artistic style presets (anime, cyberpunk, etc.)
│   │
│   ├── upscaler/
│   │   ├── realesrgan.py
│   │       # Super-resolution (1.5x, 2x, 4x)
│   │
│   ├── utils/
│   │   ├── history.py     # Prompt history & metadata saving
│   │   ├── seed.py        # Seed utilities for reproducibility
│   │   ├── logger.py      # Central logging
│   │
│   ├── models/
│   │   ├── metadata.py    # Data model for storing history entries
│
├── assets/
│   ├── samples/           # Example generated images
│   ├── lora/              # Custom LoRA models (optional)
│
├── main.py                # Entry point (launches Gradio app)
├── requirements.txt       # All dependencies (pinned)
├── LICENSE
└── README.md
```

---

# ⚙️ Installation & Setup

### Step 1 — Clone the Repo

```
git clone https://github.com/sanskarmodi8/stable-diffusion-image-generator
cd stable-diffusion-image-generator
```

### Step 2 — Create virtual environment

```
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### Step 3 — Install PyTorch (GPU)

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 4 — Install remaining dependencies

```
pip install -r requirements.txt
```

### Optional — Login to HuggingFace

```
huggingface-cli login
```

---

# ▶️ Running the App

```
python main.py
```

App will run at:

```
http://127.0.0.1:7860
```

---

# 🤝 Contributing

This project follows **strict formatting and linting standards** to ensure clean, readable, and professional-quality code.


#### 1. Install pre-commit hooks

This ensures formatting and linting run **automatically** before every commit.

```
pre-commit install
```

#### 2. Format code manually (optional)

```
black .
isort .
ruff check .
```

#### 3. Create feature branches

Follow standard naming:

```
feature/<feature-name>
fix/<bug-name>
refactor/<module>
```

#### 4. Commit messages

Use clear, conventional messages:

```
feat: add anime preset
fix: resolve img2img prompt issue
refactor: improve pipeline loading speed
docs: update readme
```

---

# 📄 License

Released under the [**MIT License**](LICENSE).

---

# ⭐ Author

**[Sanskar Modi](https://github.com/sanskarmodi8)**
AI Developer & Machine Learning Engineer