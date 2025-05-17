import streamlit as st
from openai import OpenAI
from PIL import Image
import base64
import io

# ✅ Set up your OpenAI key
client = OpenAI(api_key="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")  # Replace with your key

st.set_page_config(page_title="Smartest AI Nutrition Assistant", layout="wide")
st.title("🥗 The Smartest AI Nutrition Assistant")
st.write("Get personalized Indian meal plans, food image analysis, and expert nutrition advice — powered by GPT-4.")

# --------------------------
# 🧍 Section 1: Diet Planner
# --------------------------
st.header("🧍 Personalized Diet Plan Generator")

with st.form("diet_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=5, max_value=100, step=1)
        weight = st.number_input("Weight (in kg)")
        height = st.number_input("Height (in cm)")
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        goal = st.selectbox("Health Goal", ["Weight Loss", "Weight Gain", "Muscle Building", "Diabetes Control", "Healthy Living"])

    submitted = st.form_submit_button("Generate Diet Plan")

if submitted:
    diet_prompt = f"""
    You are a certified Indian nutritionist. Based on this user's profile:
    Age: {age}, Gender: {gender}, Weight: {weight}kg, Height: {height}cm, Goal: {goal}

    Generate a full-day Indian vegetarian meal plan including breakfast, lunch, snacks, and dinner. Mention calories and give a short explanation for each meal.
    """

    with st.spinner("Generating your personalized meal plan..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a friendly Indian dietitian giving personalized and healthy Indian meal plans."},
                    {"role": "user", "content": diet_prompt}
                ]
            )
            st.success("✅ Here's your diet plan:")
            st.markdown(response.choices[0].message.content)
        except Exception as e:
            st.error(f"❌ Error: {e}")


# -----------------------------
# 📷 Section 2: Real Image Caption + GPT-4 Turbo Nutrition Analysis
# -----------------------------
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import io

# Load BLIP model and processor (run once)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

st.header("📷 Upload Food Image for Accurate Nutrition Advice")

uploaded_img = st.file_uploader("Upload any food image (jpg/png)", type=["jpg", "jpeg", "png"])
if uploaded_img:
    st.image(uploaded_img, caption="Uploaded Food", use_column_width=True)

    # Convert image for BLIP
    img = Image.open(uploaded_img).convert('RGB')

    # 🔍 Step 1: Generate real caption
    with st.spinner("🧠 Analyzing image to understand the food..."):
        inputs = processor(img, return_tensors="pt")
        out = model_blip.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

    st.markdown(f"**🖼️ Image Caption (Detected):** `{caption}`")

    # 🔁 Step 2: Send to GPT-4 Turbo for nutrition advice
    vision_prompt = f"""
    Based on the food image, which is described as: "{caption}", please:

    1. Identify the dish as accurately as possible (e.g., Russian salad, biryani, samosa).
    2. Comment on its healthiness according to Indian dietary habits.
    3. Suggest a better alternative or preparation if needed.
    4. Mention approximate calorie range and health pros/cons.
    5. Keep the tone friendly and expert.
    """

    with st.spinner("🍽️ Generating nutrition advice..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You're a smart Indian nutritionist."},
                    {"role": "user", "content": vision_prompt}
                ],
                max_tokens=700
            )
            st.success("✅ AI Nutritionist Says:")
            st.markdown(response.choices[0].message.content)
        except Exception as e:
            st.error(f"❌ Error: {e}")

# -----------------------------
# 💬 Section 3: Chat with AI Nutritionist
# -----------------------------
st.header("💬 Ask the AI Nutritionist")

chat_input = st.text_input("Type your question (e.g., Can I eat rice at night?)")
if st.button("Get Answer") and chat_input:
    chat_prompt = f"""
    You are a smart Indian dietitian. Answer this clearly and with examples from Indian food:
    Question: {chat_input}
    """
    with st.spinner("Thinking..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You're a helpful AI nutritionist focused on Indian eating habits."},
                    {"role": "user", "content": chat_prompt}
                ]
            )
            st.success("🧠 AI Nutrition Advice:")
            st.markdown(response.choices[0].message.content)
        except Exception as e:
            st.error(f"❌ Error: {e}")
