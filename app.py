import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageFile
import numpy as np
import io
import time
import os
import pandas as pd
from datetime import datetime
import shutil
import zipfile
import random
import matplotlib.pyplot as plt
import json


ImageFile.LOAD_TRUNCATED_IMAGES = True


# Page setting
st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon="♻️",
    layout="centered")


# Custom CSS
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


# Header
st.markdown("""
<div class="header">
    <span style="font-size:36px;">♻️</span>
    <h1>AI-Based Smart Waste Management and Classification System</h1>
</div>
""", unsafe_allow_html=True)


# Create tabs
tab1, tab2 = st.tabs(["🧑 User Panel", "👨‍💼 Admin Panel"])


# Load model function
@st.cache_resource
def load_model():
    model_path = "yolo_trash_classifier.pt"
    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        return None


model = load_model()


# Disposal Guide
disposal_guide = {
    "plastic": "Please place in the recycling bin.",
    "glass": "Please place in the glass recycling bin.",
    "metal": "Please place in the metal recycling bin.",
    "organic": "Please compost this item.",
    "paper": "Please place in the paper recycling bin.",
    "unknown": "Could not classify. Please dispose of responsibly."
}


# Category colors
category_colors = {
    "plastic": {"bg": "#e3f2fd", "border": "#1565c0", "text": "#1565c0", "icon": "🔵"},
    "glass": {"bg": "#e8f5e9", "border": "#2e7d32", "text": "#2e7d32", "icon": "🟢"},
    "metal": {"bg": "#fce4ec", "border": "#b71c1c", "text": "#b71c1c", "icon": "🔴"},
    "organic": {"bg": "#f9fbe7", "border": "#827717", "text": "#827717", "icon": "🟡"},
    "paper": {"bg": "#fff3e0", "border": "#e65100", "text": "#e65100", "icon": "🟠"},
    "unknown": {"bg": "#f5f5f5", "border": "#9e9e9e", "text": "#333", "icon": "⚪"},
}


# Session state
if "history" not in st.session_state:
    st.session_state.history = []

if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False


# Load logs if exists
if "logs_df" not in st.session_state:
    if os.path.exists("classification_logs.csv"):
        st.session_state.logs_df = pd.read_csv("classification_logs.csv")
    else:
        st.session_state.logs_df = pd.DataFrame(columns=["timestamp", "image_name", "predicted_class", "actual_class", "confidence"])



# TAB 1: USER PANEL

with tab1:
    # Instructions
    with st.expander("📋 Image Upload Instructions — Click to Expand", expanded=False):
        st.markdown("""
        <div style="font-size:15px; font-weight:700; color:white; margin-bottom:12px;">
            For best results, please upload images that match the following guidelines:
        </div>
        """, unsafe_allow_html=True)


        instructions = {
            "🔵 Plastic": {"tips": ["Plastic bottles on ground or held in hand", "Crushed or used plastic bottles", "Real-world backgrounds"]},
            "🟢 Glass": {"tips": ["Glass bottles, jars, or broken glass", "Clear, green, or brown glass", "Items on a surface or held"]},
            "🔴 Metal": {"tips": ["Aluminum cans, tin cans", "Crushed or uncrushed", "Rusty or shiny both fine"]},
            "🟡 Organic": {"tips": ["Food scraps, fruit peels", "Leaves, garden waste", "Natural lighting preferred"]},
            "🟠 Paper": {"tips": ["Newspapers, cardboard", "Crumpled or flat", "Good lighting for texture"]},
        }


        for category, info in instructions.items():
            st.markdown(f"**{category}**")
            for tip in info["tips"]:
                st.markdown(f"- {tip}")
            st.markdown("---")


    # Upload section
    st.markdown("<div style='text-align:center; font-size:20px; font-weight:800; margin-bottom:20px;'>Upload or Capture an Image of Waste Item</div>", unsafe_allow_html=True)


    col1, col2 = st.columns(2)
    with col1:
        upload_btn = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])
    with col2:
        webcam_btn = st.camera_input("📷 Capture from Webcam")


    image = None
    if upload_btn:
        image = Image.open(io.BytesIO(upload_btn.getvalue())).convert("RGB")
        img_name = upload_btn.name
    elif webcam_btn:
        image = Image.open(io.BytesIO(webcam_btn.getvalue())).convert("RGB")
        img_name = "webcam_capture.jpg"


    # Prediction
    if image and model is not None:
        st.image(image, use_container_width=True)


        with st.spinner("Classifying..."):
            time.sleep(0.5)
            results = model.predict(np.array(image))
            top1 = results[0].probs.top1
            confidence = results[0].probs.top1conf.item() * 100
            
            if confidence < 35:
                predicted = "unknown"
            else:
                predicted = results[0].names[top1].lower()


        color = category_colors.get(predicted, {"bg": "#f5f5f5", "border": "#9e9e9e", "text": "#333", "icon": "⚪"})


        # Show result
        st.markdown(f"""
        <div style="background-color:{color['bg']}; border: 2px solid {color['border']}; border-radius:12px; padding:20px; text-align:center; margin-top:20px;">
            <div style="font-size:14px; font-weight:700; color:#444; margin-bottom:6px;">Classification Result:</div>
            <div style="font-size:36px;">{color['icon']}</div>
            <div style="font-size:32px; font-weight:800; color:{color['text']}; margin:6px 0;">{predicted.capitalize()} Waste</div>
            <div style="font-size:16px; color:#555;">Confidence: {confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)


        # Disposal guidance
        guide = disposal_guide.get(predicted, "Please dispose of responsibly.")
        st.markdown(f"""
        <div style="background-color:{color['bg']}; border: 2px solid {color['border']}; border-radius:10px; padding:14px 20px; margin-top:16px; font-weight:700; font-size:15px; color:{color['text']};">
            {color['icon']} <span>Disposal Guidance:</span> {guide}
        </div>
        """, unsafe_allow_html=True)


        #  actual_class selection dropdown
        all_classes_list = ["plastic", "glass", "metal", "organic", "paper"]
        actual_class = st.selectbox(
            "Verify classification (for model improvement):",
            all_classes_list,
            index=all_classes_list.index(predicted) if predicted in all_classes_list else 0
        )

        # Save to history
        st.session_state.history.append({
            "image": image.copy(),
            "label": predicted.capitalize(),
            "color": color
        })

        # Save Log Button 
        if st.button("💾 Save Log"):
            new_log = pd.DataFrame([{
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image_name": img_name,
                "predicted_class": predicted,
                "actual_class": actual_class,
                "confidence": f"{confidence:.1f}%"
            }])
            st.session_state.logs_df = pd.concat([st.session_state.logs_df, new_log], ignore_index=True)
            st.session_state.logs_df.to_csv("classification_logs.csv", index=False)
            st.success("✅ Log saved successfully!")


    elif image and model is None:
        st.error("⚠️ No trained model found. Please go to the Admin Panel and train a model first (yolo_trash_classifier.pt not found).")


    # Show history
    if st.session_state.history:
        st.markdown('<div class="section-title">Previous Classified Items:</div>', unsafe_allow_html=True)
        cols = st.columns(min(len(st.session_state.history), 4))
        for i, item in enumerate(st.session_state.history[-4:]):
            with cols[i % 4]:
                st.image(item["image"], use_container_width=True)
                c = item.get("color", {"bg": "#f5f5f5", "border": "#9e9e9e", "text": "#333", "icon": "⚪"})
                st.markdown(f'<div style="text-align:center; font-size:13px; font-weight:700; color:{c["text"]}; background:{c["bg"]}; border:1px solid {c["border"]}; border-radius:6px; padding:3px 0; margin-top:4px;">{c["icon"]} {item["label"]}</div>', unsafe_allow_html=True)


# TAB 2: ADMIN PANEL

with tab2:
    st.markdown("## 👨‍💼 Admin Panel")

    #  Admin Authentication 
    if not st.session_state.admin_authenticated:
        st.markdown("### 🔐 Admin Login")
        admin_password = st.text_input("Enter Admin Password:", type="password")
        if st.button("Login"):
            if admin_password == "admin123":
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("❌ Incorrect password. Access denied.")
        st.stop()

    st.markdown("---")

    # Logout button
    if st.button("🔓 Logout"):
        st.session_state.admin_authenticated = False
        st.rerun()

    # Section 4.1: Add dataset
    st.markdown("### 📁  Add Dataset for Training")
    st.write("Upload a zip file containing folders: plastic, glass, metal, organic, paper")
    
    dataset_zip = st.file_uploader("Upload dataset zip file", type=["zip"], key="admin_upload")
    
    if dataset_zip:
        with st.spinner("Extracting and splitting dataset..."):
          
            if os.path.exists("custom_dataset"):
                shutil.rmtree("custom_dataset")


            # Extract zip to raw folder
            os.makedirs("custom_dataset/raw", exist_ok=True)
            with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
                zip_ref.extractall("custom_dataset/raw")


            # Checking if zip already has train/val structure
            if os.path.exists("custom_dataset/raw/train") and os.path.exists("custom_dataset/raw/val"):
                # Use existing train/val structure directly
                shutil.copytree("custom_dataset/raw/train", "custom_dataset/train")
                shutil.copytree("custom_dataset/raw/val", "custom_dataset/val")
                st.success("✅ Dataset loaded directly (train/val structure detected). Now go to section 2 and click Retrain Model.")
            else:
                # Auto splitting into train/val (80/20)
                categories = ["plastic", "glass", "metal", "organic", "paper"]
                found_any = False
                for cat in categories:
                    src = None
                    for root, dirs, files in os.walk("custom_dataset/raw"):
                        if os.path.basename(root).lower() == cat:
                            src = root
                            break


                    if src is None or not os.path.exists(src):
                        continue


                    images = [f for f in os.listdir(src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if not images:
                        continue


                    found_any = True
                    random.shuffle(images)
                    split = int(len(images) * 0.8)


                    for i, img in enumerate(images):
                        folder = "train" if i < split else "val"
                        dest = f"custom_dataset/{folder}/{cat}"
                        os.makedirs(dest, exist_ok=True)
                        shutil.copy(os.path.join(src, img), os.path.join(dest, img))


                if found_any:
                    st.success("✅ Dataset extracted and split into train/val (80/20).  Now click Retrain Model.")
                else:
                    st.error("❌ No valid category folders found in zip. Make sure zip contains folders named: plastic, glass, metal, organic, paper")
    
    st.markdown("---")
    
    # Section 4.2: Retrain model
    st.markdown("### 🔄  Update Dataset & Retrain Model")
    
    if st.button("🔄 Retrain Model with New Dataset", type="primary"):
        if os.path.exists("custom_dataset"):
            st.warning("Training will take 5-10 minutes. Don't close the browser.")
            
            with st.spinner("Training in progress..."):
                try:
                    from ultralytics import YOLO as YOLOTrain
                    new_model = YOLOTrain("yolov8n-cls.pt")
                    new_model.train(data="custom_dataset", epochs=50, imgsz=224)
                    new_model.save("yolo_trash_classifier.pt")
                    
                  
                    if os.path.exists("eval_results.json"):
                        os.remove("eval_results.json")
                    
                    st.success("✅ Model retrained and saved successfully!")
                    
                    if "eval_results" in st.session_state:
                        del st.session_state.eval_results
                    st.cache_resource.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")
        else:
            st.error("No custom dataset found. Upload dataset first in section 1.")
    
    st.markdown("---")
    
    # Section 4.3: View logs
    st.markdown("### 📊 Classification Logs")
    
    if not st.session_state.logs_df.empty:
        st.dataframe(st.session_state.logs_df, use_container_width=True)
        
        # Download button for logs
        csv = st.session_state.logs_df.to_csv(index=False)
        st.download_button("📥 Download Logs as CSV", csv, "classification_logs.csv", "text/csv")
    else:
        st.info("No classification logs yet. Use the User Panel to classify some waste items.")
    
    st.markdown("---")
    
    # Section 4.4: Model evaluation report
    st.markdown("### 📈 Model Evaluation Report")
    
    if st.button("Generate Report"):
        if model is not None:
            st.markdown("#### Model Information")
            st.write(f"- Model type: YOLOv8 Classification Model")
            st.write(f"- Categories: Plastic, Glass, Metal, Organic, Paper")
            st.write(f"- Input size: 224x224 pixels")
            
            # Check if we have validation data
            val_dataset_path = None
            if os.path.exists("custom_dataset"):
                val_dataset_path = "custom_dataset"
            elif os.path.exists("dataset"):
                val_dataset_path = "dataset"
            
            if val_dataset_path:
                with st.spinner("Evaluating model on validation dataset..."):
                    try:
                     
                        if os.path.exists("eval_results.json"):
                            with open("eval_results.json") as f:
                                st.session_state.eval_results = json.load(f)
                            results_val = model.val(data="custom_dataset/val", imgsz=224, batch=16, deterministic=True, seed=42)
                        elif "eval_results" not in st.session_state:
                            results_val = model.val(data="custom_dataset/val", imgsz=224, batch=16, deterministic=True, seed=42)
                            
                            # Extract metrics safely
                            accuracy = results_val.top1 * 100 if hasattr(results_val, 'top1') else 0.0
                            
                            precision = 0.0
                            recall = 0.0
                            f1_score = 0.0
                            
                            if hasattr(results_val, 'precision') and results_val.precision is not None and len(results_val.precision) > 0:
                                valid_precision = [p for p in results_val.precision if not np.isnan(p)]
                                if valid_precision:
                                    precision = np.mean(valid_precision) * 100
                            
                            if hasattr(results_val, 'recall') and results_val.recall is not None and len(results_val.recall) > 0:
                                valid_recall = [r for r in results_val.recall if not np.isnan(r)]
                                if valid_recall:
                                    recall = np.mean(valid_recall) * 100
                            
                            if hasattr(results_val, 'f1') and results_val.f1 is not None and len(results_val.f1) > 0:
                                valid_f1 = [f for f in results_val.f1 if not np.isnan(f)]
                                if valid_f1:
                                    f1_score = np.mean(valid_f1) * 100
                            
                            if precision == 0.0 and hasattr(results_val, 'confusion_matrix'):
                                try:
                                    cm = results_val.confusion_matrix.matrix
                                    tp = np.diag(cm)
                                    fp = np.sum(cm, axis=0) - tp
                                    fn = np.sum(cm, axis=1) - tp
                                    
                                    precisions = tp / (tp + fp + 1e-9)
                                    recalls = tp / (tp + fn + 1e-9)
                                    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
                                    
                                    precision = np.mean(precisions) * 100
                                    recall = np.mean(recalls) * 100
                                    f1_score = np.mean(f1s) * 100
                                except:
                                    pass
                            
                            st.session_state.eval_results = {
                                "accuracy": accuracy,
                                "precision": precision,
                                "recall": recall,
                                "f1_score": f1_score,
                            }
                            
                            with open("eval_results.json", "w") as f:
                                json.dump(st.session_state.eval_results, f)
                        else:
                            results_val = model.val(data="custom_dataset/val", imgsz=224, batch=16, deterministic=True, seed=42)

                        results_cached = st.session_state.eval_results
                        accuracy = results_cached["accuracy"]
                        precision = results_cached["precision"]
                        recall = results_cached["recall"]
                        f1_score = results_cached["f1_score"]


                        # Performance Metrics
                        st.markdown("#### 📊 Performance Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("🎯 Accuracy", f"{accuracy:.1f}%")
                        with col2:
                            st.metric("⚡ Precision", f"{precision:.1f}%")
                        with col3:
                            st.metric("📊 Recall", f"{recall:.1f}%")
                        with col4:
                            st.metric("🎖️ F1 Score", f"{f1_score:.1f}%")
                        
                        # Confusion Matrix
                        st.markdown("#### 🔢 Confusion Matrix")
                        st.write("Shows actual vs predicted classifications on validation dataset")
                        
                        try:
                            if hasattr(results_val, 'confusion_matrix') and results_val.confusion_matrix is not None:
                                cm = results_val.confusion_matrix.matrix
                                cm_int = cm.astype(int)
                                
                                num_classes = cm_int.shape[0]
                                class_names_cm = ["plastic", "glass", "metal", "organic", "paper", "unknown"]
                                class_names_cm = class_names_cm[:num_classes]
                                
                                fig, ax = plt.subplots(figsize=(8, 6))
                                im = ax.imshow(cm_int, interpolation='nearest', cmap='Blues')
                                ax.figure.colorbar(im, ax=ax)
                                
                                ax.set(xticks=np.arange(num_classes),
                                       yticks=np.arange(num_classes),
                                       xticklabels=class_names_cm,
                                       yticklabels=class_names_cm,
                                       xlabel='Predicted Class',
                                       ylabel='Actual Class')
                                
                                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                                
                                for i in range(num_classes):
                                    for j in range(num_classes):
                                        ax.text(j, i, format(cm_int[i, j], 'd'),
                                                ha="center", va="center",
                                                color="white" if cm_int[i, j] > cm_int.max() / 2 else "black")
                                
                                ax.set_title('Confusion Matrix (from validation dataset)')
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                                
                                st.caption("Rows = Actual classes, Columns = Predicted classes")
                                st.caption("Diagonal cells = Correct predictions, Off-diagonal = Misclassifications")
                            else:
                                st.info("Confusion matrix not available from validation dataset")
                        
                        except Exception as e:
                            st.error(f"Could not generate confusion matrix: {e}")
                        
                        # Overall summary
                        st.markdown("#### 📝 Model Summary")
                        if accuracy > 90:
                            st.success(f"✅ **Excellent Performance!** The model achieves {accuracy:.1f}% accuracy with balanced precision and recall across all waste categories.")
                        elif accuracy > 75:
                            st.warning(f"⚠️ **Good Performance.** Model accuracy is {accuracy:.1f}%. Consider adding more diverse training data for improvement.")
                        else:
                            st.error(f"❌ **Needs Improvement.** Model accuracy is {accuracy:.1f}%. Please add more training data and retrain.")
                        
                        # Dataset information
                        st.markdown("#### 📁 Dataset Summary")
                        if os.path.exists("custom_dataset"):
                            total_images = 0
                            for cat in ["plastic", "glass", "metal", "organic", "paper"]:
                                cat_path = f"custom_dataset/train/{cat}"
                                if os.path.exists(cat_path):
                                    num = len([f for f in os.listdir(cat_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
                                    total_images += num
                                    st.write(f"- **{cat.capitalize()}:** {num} training images")
                            st.write(f"**Total training images:** {total_images}")
                            st.write(f"**Train/Validation split:** 80/20 (standard YOLO split)")
                        else:
                            st.write("- Using default pre-trained model")
                            st.write("- Upload custom dataset and retrain for detailed evaluation")
                    
                    except Exception as e:
                        st.error(f"Evaluation failed: {str(e)}")
                        st.info("Make sure your dataset folder has the correct structure: custom_dataset/train/ and custom_dataset/val/ with class subfolders")
            else:
                st.warning("No validation dataset found. Please:")
                st.write("1. Upload a dataset zip file in section 1")
                st.write("2. Click 'Retrain Model' in section 2")
                st.write("3. Then generate this report again")
        else:
            st.error("⚠️ No trained model found. Please upload a dataset and retrain the model first (yolo_trash_classifier.pt not found).")
    
    st.markdown("---")
    st.markdown("### ℹ️ Admin Instructions")
    st.write("1. Upload a zip file with folders named: plastic, glass, metal, organic, paper")
    st.write("2. Click 'Retrain Model' to train on new data")
    st.write("3. View logs to monitor performance")
    st.write("4. Generate report to see model evaluation")


# Footer
st.markdown("""
<div style="text-align:center; margin-top:40px; font-size:13px; color:#888; font-weight:600;">
    ♻️ Smart Waste Classifier — Helping build a cleaner world<br>
    <span style="font-size:12px; color:#aaa;">Developed by: Iqra Jamil &nbsp;|&nbsp; Student ID: Bc220200221 &nbsp;|&nbsp; Final Year Project 2025</span>
</div>
""", unsafe_allow_html=True)
