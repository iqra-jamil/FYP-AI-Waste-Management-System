import streamlit as st # type: ignore
from ultralytics import YOLO # type: ignore
from PIL import Image, ImageFile # type: ignore
import numpy as np # type: ignore
import time
import io

ImageFile.LOAD_TRUNCATED_IMAGES = True
#page setting
st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon="♻️", 
    layout="centered")
#Custom design using css
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
* { font-family: 'Nunito', sans-serif; }
.header {
    background-color: #2e7d32;
    padding: 18px 24px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 30px;
}
.header h1 {
    color: white;
    font-size: 22px;
    font-weight: 800;
    margin: 0;
}
.section-title {
    font-size: 18px;
    font-weight: 800;
    color: #fff;
    margin-top: 30px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)
#displaying title + icon (header)
st.markdown("""
<div class="header">
    <span style="font-size:36px;">♻️</span>
    <h1>AI-Based Smart Waste Management and Classification System</h1>
</div>
""", unsafe_allow_html=True)
#Load Model
@st.cache_resource
def load_model():
    return YOLO('yolo_trash_classifier.pt')

model = load_model()
#Disposal Giude Dictionary
disposal_guide = {
    "plastic": "Please place in the recycling bin.",
    "glass": "Please place in the glass recycling bin.",
    "metal": "Please place in the metal recycling bin.",
    "organic": "Please compost this item.",
    "paper": "Please place in the paper recycling bin.",
    "unknown": "Could not classify. Please dispose of responsibly."
}
#color styling for each category
category_colors = {
    "plastic": {"bg": "#e3f2fd", "border": "#1565c0", "text": "#1565c0", "icon": "🔵"},
    "glass":   {"bg": "#e8f5e9", "border": "#2e7d32", "text": "#2e7d32", "icon": "🟢"},
    "metal":   {"bg": "#fce4ec", "border": "#b71c1c", "text": "#b71c1c", "icon": "🔴"},
    "organic": {"bg": "#f9fbe7", "border": "#827717", "text": "#827717", "icon": "🟡"},
    "paper":   {"bg": "#fff3e0", "border": "#e65100", "text": "#e65100", "icon": "🟠"},
    "unknown": {"bg": "#f5f5f5", "border": "#9e9e9e", "text": "#333", "icon": "⚪"},
}

#Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

#Instructions Section 
with st.expander("📋 Image Upload Instructions — Click to Expand", expanded=False):
    st.markdown("""
    <div style="font-size:15px; font-weight:700; color:white; margin-bottom:12px;">
        For best results, please upload images that match the following guidelines for each waste type:
    </div>
    """, unsafe_allow_html=True)

    instructions = {
        "🔵 Plastic": {
            "color": "#e3f2fd",
            "border": "#1565c0",
            "text": "#1565c0",
            "tips": [
                "Plastic bottles lying on ground or held in hand",
                "Crushed, dirty, or used plastic bottles",
                "Real-world backgrounds (floor, outdoor, etc.)",
                "Avoid pure white studio-style product shots",
                "Horizontal or angled orientations work best",
            ]
        },
        "🟢 Glass": {
            "color": "#e8f5e9",
            "border": "#2e7d32",
            "text": "#2e7d32",
            "tips": [
                "Glass bottles, jars, or broken glass pieces",
                "Clear, green, or brown glass items",
                "Items placed on a surface or held in hand",
                "Avoid reflective backgrounds that hide the glass shape",
            ]
        },
        "🔴 Metal": {
            "color": "#fce4ec",
            "border": "#b71c1c",
            "text": "#b71c1c",
            "tips": [
                "Aluminum cans, tin cans, or metal scraps",
                "Crushed or uncrushed cans both work",
                "Rusty or shiny metal items are fine",
                "Avoid images with multiple mixed items",
            ]
        },
        "🟡 Organic": {
            "color": "#f9fbe7",
            "border": "#827717",
            "text": "#827717",
            "tips": [
                "Food scraps, fruit peels, vegetable waste",
                "Leaves, garden waste, or leftover food",
                "Natural lighting preferred",
                "Single item or small pile works best",
            ]
        },
        "🟠 Paper": {
            "color": "#fff3e0",
            "border": "#e65100",
            "text": "#e65100",
            "tips": [
                "Newspapers, cardboard boxes, paper bags",
                "Crumpled or flat paper both acceptable",
                "Avoid images where paper is mixed with other waste",
                "Good lighting to distinguish paper texture",
            ]
        },
    }

    for category, info in instructions.items():
        st.markdown(f"""
        <div style="background-color:{info['color']}; border: 2px solid {info['border']}; 
             border-radius:10px; padding:14px 18px; margin-bottom:12px;">
            <div style="font-size:16px; font-weight:800; color:{info['text']}; margin-bottom:8px;">{category}</div>
            <ul style="margin:0; padding-left:18px; color:#444; font-size:14px; font-weight:600;">
                {''.join(f'<li>{tip}</li>' for tip in info['tips'])}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color:#f5f5f5; border:2px solid #9e9e9e; border-radius:10px; 
         padding:12px 18px; font-size:14px; font-weight:600; color:#555;">
        ⚠️ <strong>General Tips:</strong> Use clear, well-lit images. Make sure the waste item is 
        the main focus of the photo. Avoid very dark, blurry, or heavily filtered images for best accuracy.
    </div>
    """, unsafe_allow_html=True)


#upload and camera Input
st.markdown("<div style='text-align:center; font-size:20px; font-weight:800; margin-bottom:20px;'>Upload or Capture an Image of Waste Item</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    upload_btn = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])
with col2:
    webcam_btn = st.camera_input("📷 Capture from Webcam")

#image processing
image = None
if upload_btn:
    image = Image.open(io.BytesIO(upload_btn.getvalue())).convert("RGB")
elif webcam_btn:
    image = Image.open(io.BytesIO(webcam_btn.getvalue())).convert("RGB")

#Model Prediction
if image:
    st.image(image, use_container_width=True)

    with st.spinner("Classifying..."):
        time.sleep(0.5)
        results = model.predict(np.array(image))
        top1 = results[0].probs.top1
        confidence = results[0].probs.top1conf.item() * 100 
        #unknown or top predicted class
        if confidence < 35:
            predicted = "unknown"
        else:
            predicted = results[0].names[top1].lower()

    color = category_colors.get(predicted, {"bg": "#f5f5f5", "border": "#9e9e9e", "text": "#333", "icon": "⚪"})

#Showing result card
    st.markdown(f"""
    <div style="background-color:{color['bg']}; border: 2px solid {color['border']}; border-radius:12px; padding:20px; text-align:center; margin-top:20px;">
        <div style="font-size:14px; font-weight:700; color:#444; margin-bottom:6px;">Classification Result:</div>
        <div style="font-size:36px;">{color['icon']}</div>
        <div style="font-size:32px; font-weight:800; color:{color['text']}; margin:6px 0;">{predicted.capitalize()} Waste</div>
        <div style="font-size:16px; color:#555;">Confidence: {confidence:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    #Disposal Guidence box
    guide = disposal_guide.get(predicted, "Please dispose of responsibly.")
    st.markdown(f"""
    <div style="background-color:{color['bg']}; border: 2px solid {color['border']}; border-radius:10px; padding:14px 20px; margin-top:16px; font-weight:700; font-size:15px; color:{color['text']};">
        {color['icon']} <span>Disposal Guidance:</span> {guide}
    </div>
    """, unsafe_allow_html=True)

    #Save to history
    st.session_state.history.append({
        "image": image.copy(), 
        "label": predicted.capitalize(),
          "color": color
        })

        
#showing the previous images
if st.session_state.history:
    st.markdown('<div class="section-title">Previous Classified Items:</div>', unsafe_allow_html=True)
    cols = st.columns(min(len(st.session_state.history), 4))
    for i, item in enumerate(st.session_state.history[-4:]):
        with cols[i % 4]:
            st.image(item["image"], use_container_width=True)
            c = item.get("color", {"bg": "#f5f5f5", "border": "#9e9e9e", "text": "#333", "icon": "⚪"})
            st.markdown(f'<div style="text-align:center; font-size:13px; font-weight:700; color:{c["text"]}; background:{c["bg"]}; border:1px solid {c["border"]}; border-radius:6px; padding:3px 0; margin-top:4px;">{c["icon"]} {item["label"]}</div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; margin-top:40px; font-size:13px; color:#888; font-weight:600;">
    ♻️ Smart Waste Classifier — Helping build a cleaner world<br>
    <span style="font-size:12px; color:#aaa;">Developed by: Iqra jamil &nbsp;|&nbsp; Student ID: Bc220200221 &nbsp;|&nbsp; Final Year Project 2025</span>
</div>
""", unsafe_allow_html=True)