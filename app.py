import os
import io

import torch
import streamlit as st
from PIL import Image
from torchvision import transforms

from model import build_mobilenet

MODEL_OPTIONS = {
    "Standard Model (alpha = 1.00)": {"alpha": 1.0, "path": "mobilenet_alpha_1.0.pth"},
    "Lightweight Model (alpha = 0.35)": {"alpha": 0.35, "path": "mobilenet_alpha_0.35.pth"},
}

INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

def get_class_names():
    train_dir = "./my_custom_dataset/train"
    if not os.path.isdir(train_dir):
        return []

    return sorted(
        entry.name
        for entry in os.scandir(train_dir)
        if entry.is_dir()
    )

@st.cache_resource(show_spinner="Loading model weights...")
def load_model(alpha, weights_path, num_classes):
    if not os.path.isfile(weights_path):
        return None

    model = build_mobilenet(alpha=alpha, num_classes=num_classes)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    return model

def file_size_mb(path):
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 ** 2)
    return 0.0

def predict(model, image, class_names):
    image = image.convert("RGB")

    tensor = INFER_TRANSFORM(image)
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = probs.max(dim=1)

    label = class_names[pred_idx.item()] if class_names else str(pred_idx.item())
    confidence = conf.item() * 100.0

    return label, confidence

def main():
    st.set_page_config(
        page_title="Edge AI Image Classifier",
        page_icon="AI",
        layout="wide",
    )

    st.title("Edge AI Image Classifier")
    st.markdown(
        "MobileNetV2 inference dashboard - compare a standard model against "
        "an ultra-lightweight edge-optimized variant."
    )

    st.divider()

    with st.sidebar:
        st.header("Model Selection")

        selected_label = st.selectbox(
            label="Choose a model variant",
            options=list(MODEL_OPTIONS.keys()),
            help="Switch between the two trained MobileNetV2 checkpoints.",
        )

        cfg = MODEL_OPTIONS[selected_label]
        alpha = cfg["alpha"]
        path = cfg["path"]

        st.markdown("---")
        st.subheader("Model Footprint")

        size = file_size_mb(path)

        if size > 0:
            st.metric(label="Checkpoint size", value=f"{size:.2f} MB")
        else:
            st.warning(f"{path} not found. Run train.py first.")

        if alpha == 1.0:
            lite_size = file_size_mb(
                MODEL_OPTIONS["Lightweight Model (alpha = 0.35)"]["path"]
            )

            if lite_size > 0:
                savings = (1 - lite_size / size) * 100 if size > 0 else 0

                st.markdown(
                    f"The lightweight model (alpha=0.35) is "
                    f"{savings:.0f}% smaller ({lite_size:.2f} MB vs {size:.2f} MB). "
                    f"It reduces channel widths and computation at the cost of some accuracy."
                )
        else:
            std_size = file_size_mb(
                MODEL_OPTIONS["Standard Model (alpha = 1.00)"]["path"]
            )

            if std_size > 0:
                savings = (1 - size / std_size) * 100 if std_size > 0 else 0

                st.markdown(
                    f"This lightweight model (alpha=0.35) is "
                    f"{savings:.0f}% smaller than the standard model."
                )

        st.markdown("---")
        st.caption("Built with PyTorch, MobileNetV2 and Streamlit")

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.subheader("Upload an Image")

        uploaded_file = st.file_uploader(
            label="Drop any image file here",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Supported formats: JPEG, PNG, BMP, WebP",
        )

        if uploaded_file is not None:
            pil_image = Image.open(io.BytesIO(uploaded_file.read()))
            st.image(pil_image, caption="Uploaded image", use_container_width=True)

    with col_result:
        st.subheader("Prediction")

        if uploaded_file is None:
            st.info("Upload an image on the left to run inference.")
        else:
            class_names = get_class_names()

            if not class_names:
                st.error(
                    "Could not find class folders under ./my_custom_dataset/train/"
                )
                return

            num_classes = len(class_names)

            model = load_model(alpha, path, num_classes)

            if model is None:
                st.error(
                    f"Weights file {path} not found. Train the model first."
                )
                return

            with st.spinner("Running inference..."):
                label, confidence = predict(model, pil_image, class_names)

            st.success(f"Predicted class: {label}")

            st.metric(
                label="Confidence",
                value=f"{confidence:.1f}%"
            )

            st.progress(int(confidence))

            st.markdown("---")
            st.markdown("All classes detected in dataset:")
            st.write(", ".join(class_names))

            st.markdown("---")
            st.markdown(
                f"Inference ran on: {selected_label} | "
                f"Width multiplier: {alpha} | "
                f"Checkpoint size: {size:.2f} MB"
            )

if __name__ == "__main__":
    main()