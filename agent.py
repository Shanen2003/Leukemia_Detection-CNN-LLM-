import ollama
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import re
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import streamlit as st
from functools import lru_cache

@lru_cache(maxsize=1)
def load_model():
    return tf.keras.models.load_model('./Model_Basic.keras')

# @lru_cache(maxsize=1)  
# def load_playbook():
#     with open("playbook.md", "r") as f:
#         return f.read()

model = load_model()
# playbook = load_playbook()

# Global df — gets replaced when uploads come in
df = pd.DataFrame()

SYSTEM_PROMPT = """
You are a data routing assistant. Your ONLY job is to return a JSON object.
You must ALWAYS return valid JSON and nothing else. No explanations, no disclaimers, no text.

Given a user question, return this exact JSON format:

{
    "section": "section_name_here",
    "placeholders": {
        "PATIENT_ID": 86,
        "TOP_N": 5
    }
}

Rules:
- ALWAYS return JSON. Never return text or explanations.
- If the user mentions a patient number, use it as PATIENT_ID
- If the user does NOT mention a patient number, use the MOST RECENTLY UPLOADED patient ID
- Default TOP_N to 5 if not specified
- For compare_patients, COMPARE_PATIENT_ID should be the other patient

Available sections:
- predict_patient
- show_top_cancer_cells
- show_top_healthy_cells
- show_gradcam
- compare_patients
"""

NARRATION_PROMPT = """
You are a friendly and conversational medical AI assistant helping a doctor understand leukemia cell predictions at NewYork-Presbyterian Hospital.
Given the results of an analysis, explain what happened in plain English in 2-3 sentences.
Be warm, clear, professional, and avoid overly technical language.
Do not make definitive medical diagnoses — frame it as model predictions to assist the physician.

Important context for predictions:
- The model analyses each individual cell image and assigns a probability of leukemia
- The probabilities are averaged across ALL cells for the patient
- If the average is ABOVE 50% the patient is flagged as Leukemia Detected
- If the average is BELOW 50% the patient is flagged as No Leukemia Detected
- Always mention the average confidence and the final prediction in your response
"""

def get_patient_predictions(patient_id):
    cache_key = str(patient_id)
    if hasattr(get_patient_predictions, '_cache') and cache_key in get_patient_predictions._cache:
        return get_patient_predictions._cache[cache_key]
    
    df_patient = df[df["Patient_New_ID"] == patient_id].copy().reset_index(drop=True)
    real_count = len(df_patient)
    df_patient["Cancer"] = "0"
    
    # Add one dummy row for each class so generator sees 2 classes
    dummy_0 = df_patient.iloc[0:1].copy()
    dummy_0["Cancer"] = "0"
    dummy_1 = df_patient.iloc[0:1].copy()
    dummy_1["Cancer"] = "1"
    
    df_with_dummy = pd.concat([df_patient, dummy_0, dummy_1], ignore_index=True)

    datagen = ImageDataGenerator(rescale=1./255)
    pred_gen = datagen.flow_from_dataframe(
        dataframe=df_with_dummy,
        x_col="FilePath",
        y_col="Cancer",
        target_size=(64, 64),
        batch_size=32,
        class_mode="binary",
        shuffle=False
    )
    
    all_preds = model.predict(pred_gen, verbose=0)
    
    # Only take the REAL rows — explicitly slice by real_count
    df_patient["Predicted_Prob"] = all_preds[:real_count]
    
    print(f"Patient {patient_id} avg prob: {df_patient['Predicted_Prob'].mean():.4f} across {real_count} real cells")
    
    if not hasattr(get_patient_predictions, '_cache'):
        get_patient_predictions._cache = {}
    get_patient_predictions._cache[cache_key] = df_patient
    
    return df_patient

def fig_predict_patient(patient_id):
    df_patient = get_patient_predictions(patient_id)
    avg_prob = df_patient["Predicted_Prob"].mean()
    prediction = "Leukemia Detected" if avg_prob > 0.5 else "No Leukemia Detected"
    result = {
        "patient_id": patient_id,
        "prediction": prediction,
        "confidence": float(avg_prob),
        "num_images": len(df_patient),
        "cells_above_50": int((df_patient["Predicted_Prob"] > 0.5).sum()),
        "cells_below_50": int((df_patient["Predicted_Prob"] <= 0.5).sum())
    }
    return None, result


def fig_show_top_cancer_cells(patient_id, top_n=5):
    df_patient = get_patient_predictions(patient_id)
    top_cells = df_patient.nlargest(top_n, "Predicted_Prob").reset_index(drop=True)

    fig, axes = plt.subplots(1, top_n, figsize=(4 * top_n, 4))
    fig.suptitle(f'Top {top_n} Most Likely Leukemia Cells — Patient {patient_id}',
                 fontsize=14, fontweight='bold', color='#1a1a2e')
    if top_n == 1:
        axes = [axes]

    for i, row in top_cells.iterrows():
        original = mpimg.imread(row["FilePath"])
        axes[i].imshow(original)
        axes[i].set_title(f"Confidence: {row['Predicted_Prob']:.2%}", fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    result = {
        "patient_id": patient_id,
        "top_n": top_n,
        "confidences": top_cells["Predicted_Prob"].tolist()
    }
    return fig, result


def fig_show_top_healthy_cells(patient_id, top_n=5):
    df_patient = get_patient_predictions(patient_id)
    healthy_cells = df_patient.nsmallest(top_n, "Predicted_Prob").reset_index(drop=True)

    fig, axes = plt.subplots(1, top_n, figsize=(4 * top_n, 4))
    fig.suptitle(f'Top {top_n} Most Likely Healthy Cells — Patient {patient_id}',
                 fontsize=14, fontweight='bold', color='#1a1a2e')
    if top_n == 1:
        axes = [axes]

    for i, row in healthy_cells.iterrows():
        original = mpimg.imread(row["FilePath"])
        axes[i].imshow(original)
        axes[i].set_title(f"Confidence: {row['Predicted_Prob']:.2%}", fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    result = {
        "patient_id": patient_id,
        "top_n": top_n,
        "confidences": healthy_cells["Predicted_Prob"].tolist()
    }
    return fig, result


def fig_show_gradcam(patient_id, top_n=5):
    from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
    from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
    from tf_keras_vis.utils.scores import BinaryScore

    df_patient = get_patient_predictions(patient_id)
    top_cells = df_patient.nlargest(top_n, "Predicted_Prob").reset_index(drop=True)

    def get_last_conv_layer(m):
        for layer in reversed(m.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name

    last_conv = get_last_conv_layer(model)
    gradcam = GradcamPlusPlus(model, model_modifier=ReplaceToLinear(), clone=True)

    fig, axes = plt.subplots(top_n, 2, figsize=(10, 4 * top_n))
    fig.suptitle(f'Grad-CAM++ Heatmaps — Patient {patient_id}',
                 fontsize=14, fontweight='bold', color='#1a1a2e')

    if top_n == 1:
        axes = [axes]

    for i, row in top_cells.iterrows():
        img_array = img_to_array(load_img(row["FilePath"], target_size=(64, 64))) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        score = BinaryScore(True)
        cam = gradcam(score, img_array, penultimate_layer=last_conv)
        heatmap = cam[0]
        original = mpimg.imread(row["FilePath"])

        axes[i][0].imshow(original)
        axes[i][0].set_title(f"Original | {row['Predicted_Prob']:.2%}", fontsize=10)
        axes[i][0].axis('off')

        axes[i][1].imshow(original)
        axes[i][1].imshow(heatmap, cmap='jet', alpha=0.5)
        axes[i][1].set_title("Grad-CAM++ Heatmap", fontsize=10)
        axes[i][1].axis('off')

    plt.tight_layout()
    result = {
        "patient_id": patient_id,
        "top_n": top_n,
        "confidences": top_cells["Predicted_Prob"].tolist()
    }
    return fig, result


def fig_compare_patients(patient_id_1, patient_id_2):
    df_p1 = get_patient_predictions(patient_id_1)
    df_p2 = get_patient_predictions(patient_id_2)

    avg_p1 = df_p1["Predicted_Prob"].mean()
    avg_p2 = df_p2["Predicted_Prob"].mean()

    top_p1 = df_p1.nlargest(3, "Predicted_Prob").reset_index(drop=True)
    top_p2 = df_p2.nlargest(3, "Predicted_Prob").reset_index(drop=True)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f'Patient {patient_id_1} vs Patient {patient_id_2}',
                 fontsize=14, fontweight='bold', color='#1a1a2e')

    for i in range(3):
        original = mpimg.imread(top_p1.iloc[i]["FilePath"])
        axes[0, i].imshow(original)
        axes[0, i].set_title(f"P{patient_id_1} | {top_p1.iloc[i]['Predicted_Prob']:.2%}", fontsize=10)
        axes[0, i].axis('off')

        original = mpimg.imread(top_p2.iloc[i]["FilePath"])
        axes[1, i].imshow(original)
        axes[1, i].set_title(f"P{patient_id_2} | {top_p2.iloc[i]['Predicted_Prob']:.2%}", fontsize=10)
        axes[1, i].axis('off')

    plt.tight_layout()
    result = {
        "patient_id_1": patient_id_1,
        "patient_id_2": patient_id_2,
        "avg_confidence_1": float(avg_p1),
        "avg_confidence_2": float(avg_p2),
        "prediction_1": "Leukemia Detected" if avg_p1 > 0.5 else "No Leukemia Detected",
        "prediction_2": "Leukemia Detected" if avg_p2 > 0.5 else "No Leukemia Detected"
    }
    return fig, result


def narrate_result(user_question, result):
    # Build a clean summary to pass to the LLM
    if "confidence" in result:
        summary = f"Overall average probability across ALL cells: {result['confidence']:.4f} ({result['confidence']:.1%}). Final prediction: {result['prediction']}. Total cells analysed: {result['num_images']}."
    elif "confidences" in result:
        overall_avg = sum(result["confidences"]) / len(result["confidences"])
        summary = f"These are the TOP {result['top_n']} highest confidence cells only — NOT the overall patient prediction. Their individual confidences: {[f'{c:.1%}' for c in result['confidences']]}."
    elif "avg_confidence_1" in result:
        summary = f"Patient {result['patient_id_1']} overall average: {result['avg_confidence_1']:.1%} — {result['prediction_1']}. Patient {result['patient_id_2']} overall average: {result['avg_confidence_2']:.1%} — {result['prediction_2']}."
    else:
        summary = str(result)

    response = ollama.chat(
        model='phi3:mini',
        messages=[
            {'role': 'system', 'content': NARRATION_PROMPT},
            {'role': 'user', 'content': f"User asked: {user_question}\n\nAnalysis results: {summary}"}
        ]
    )
    return response['message']['content']


def ask_agent(user_question, uploaded_patients=None):
    global df
    if uploaded_patients:
        df = pd.concat(uploaded_patients.values(), ignore_index=True)
        available_ids = list(uploaded_patients.keys())
        latest_patient = available_ids[-1]
    else:
        available_ids = [86]
        latest_patient = 86

    # Always guess section from keywords first — don't trust Phi-3 for routing
    question_lower = user_question.lower()

    if any(w in question_lower for w in ["heatmap", "gradcam", "why", "highlight", "where"]):
        section = "show_gradcam"
    elif any(w in question_lower for w in ["compare", "vs", "versus", "difference between"]):
        section = "compare_patients"
    elif any(w in question_lower for w in ["healthy", "normal", "negative", "not cancer", "no cancer"]):
        section = "show_top_healthy_cells"
    elif any(w in question_lower for w in ["top cancer", "top cells", "most likely cancer", "highest"]):
        section = "show_top_cancer_cells"
    else:
        # Everything else — "is this patient likely", "predict", "does this patient have" etc
        section = "predict_patient"

    # Extract patient ID from question if mentioned, otherwise use latest
    patient_id = latest_patient
    for pid in available_ids:
        if str(pid) in question_lower:
            patient_id = pid
            break

    # Extract top_n if mentioned
    top_n = 5
    top_n_match = re.search(r'\b(\d+)\s*(cells?|images?|examples?)\b', question_lower)
    if top_n_match:
        top_n = int(top_n_match.group(1))

    # Extract compare ID if mentioned
    compare_id = available_ids[0] if len(available_ids) > 1 and available_ids[0] != patient_id else (available_ids[1] if len(available_ids) > 1 else patient_id)

    print(f"=== ROUTING === section={section} | patient_id={patient_id} | top_n={top_n}")

    try:
        if section == "predict_patient":
            fig, result = fig_predict_patient(patient_id)
        elif section == "show_top_cancer_cells":
            fig, result = fig_show_top_cancer_cells(patient_id, top_n)
        elif section == "show_top_healthy_cells":
            fig, result = fig_show_top_healthy_cells(patient_id, top_n)
        elif section == "show_gradcam":
            fig, result = fig_show_gradcam(patient_id, top_n)
        elif section == "compare_patients":
            fig, result = fig_compare_patients(patient_id, compare_id)
        else:
            return None, None, "I'm not sure how to help with that. Try asking about a patient's cells or prediction."
    except Exception as e:
        return None, None, f"Something went wrong running the analysis: {e}"

    narration = narrate_result(user_question, result)
    return fig, result, narration