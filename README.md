# Text-to-Image Generator using Stable Diffusion and Language Translation

This is a Google Colab-compatible Python project that translates input text into English (if needed) and generates an image using the [Stable Diffusion v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1) model from Hugging Face. It is optimized for both CPU and GPU and includes support for multiple languages via the `deep_translator` package.

---

## 🚀 Features

- 🔤 **Language Translation** using Google Translate API (`deep_translator`)
- 🎨 **Text-to-Image Generation** using Stable Diffusion 2.1
- ⚡ **GPU Acceleration** if available (uses `float16`)
- 💾 **Downloadable Image Output** via Google Colab
- 🧠 Configurable parameters for reproducibility and control

---

## 📦 Installation

Run the following commands in a **Google Colab** notebook cell:

```python
!pip install deep_translator -q
!pip install --upgrade diffusers transformers -q
```

---

## 🧠 How It Works

1. **Translate** non-English text using `deep_translator`.
2. **Initialize** the Stable Diffusion model using Hugging Face.
3. **Generate** a high-resolution image from the translated prompt.
4. **Display** and optionally download the image.

---

## 🧰 Dependencies

- `torch`
- `diffusers`
- `transformers`
- `deep_translator`
- `matplotlib`
- `ipywidgets` (for download button in Colab)
- `google.colab` (for file download in Colab)

---

## ⚙️ Configuration

```python
class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2-1"
    image_gen_size = (900, 900)
    image_gen_guidance_scale = 9
```

You can change these values to adjust model behavior.

---

## 🖼️ Example Usage

```python
# Replace with your Hugging Face token
auth_token = 'your_hugging_face_auth_token'

model = initialize_model(CFG.image_gen_model_id, auth_token)

if model:
    translation = get_translation("person on a table with dog", "en")
    print(f"Translated prompt: {translation}")
    image = generate_image(translation, model)

    if image:
        plt.imshow(image)
        plt.axis('off')
        plt.show()
```

---

## 🔐 Hugging Face Token

To access the model, you need an access token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Pass it to the `initialize_model` function.

---

## 📤 Output

- Saved as: `/content/generated_image.png`
- Downloadable directly via a button in the notebook

---

## 📄 License

This project is for educational and research purposes. Refer to [Stability AI’s license](https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/LICENSE.md) for usage restrictions on the model.

---
