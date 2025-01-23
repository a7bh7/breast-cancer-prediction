import streamlit as st
import numpy as np
import pickle

# تحميل النموذج والمعلومات

with open('models/breast_cancer_model.pkl', 'rb') as file:
    model, scaler, feature_names = pickle.load(file)

# عنوان التطبيق
st.title("تطبيق توقع الإصابة بسرطان الثدي")
st.write("قم بإدخال القيم الطبية للحصول على التوقع.")

# إدخال القيم من المستخدم
user_input = []
for feature in feature_names:
    value = st.number_input(f"{feature}", step=0.1)
    user_input.append(value)

# التنبؤ
if st.button("توقع"):
    input_array = np.array(user_input).reshape(1, -1)
    input_array = scaler.transform(input_array)  # تطبيق القياس الموحّد
    prediction = model.predict(input_array)[0]

    if prediction == 1:
        st.error("النتيجة: مصاب بسرطان الثدي.")
    else:
        st.success("النتيجة: غير مصاب بسرطان الثدي.")
