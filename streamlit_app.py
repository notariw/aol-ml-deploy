import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load pipeline
model = joblib.load("pipeline_diabetes.pkl")
# Load feature names (ini akan tetap berisi nama fitur transformasi)
feature_names_transformed = joblib.load("feature_names.pkl")

st.set_page_config(layout="centered", initial_sidebar_state="auto", page_title="Prediksi Risiko Diabetes")

st.title("📈 Prediksi Risiko Diabetes 📈")
st.markdown("Aplikasi ini memprediksi risiko diabetes berdasarkan data yang Anda masukkan.")

st.subheader("Input Data Pasien")

# Define original 8 numerical columns for plotting
original_numerical_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Menggunakan st.columns untuk membagi input menjadi 2 kolom
col1, col2 = st.columns(2)

# Input dari pengguna
with st.form("diabetes_form"):
    with col1: # Kolom Kiri
        pregnancies = st.number_input("Jumlah Kehamilan (Pregnancies)", min_value=0, value=0, help="Jumlah kehamilan sebelumnya")
        glucose = st.number_input("Kadar Glukosa (Glucose)", min_value=0.0, value=120.0, help="Konsentrasi glukosa plasma 2 jam dalam tes toleransi glukosa oral")
        blood_pressure = st.number_input("Tekanan Darah (Blood Pressure)", min_value=0.0, value=70.0, help="Tekanan darah diastolik (mm Hg)")
        skin_thickness = st.number_input("Ketebalan Kulit (Skin Thickness)", min_value=0.0, value=20.0, help="Ketebalan lipatan kulit trisep (mm)")

    with col2: # Kolom Kanan
        insulin = st.number_input("Insulin", min_value=0.0, value=80.0, help="Tingkat insulin serum 2 jam (mu U/ml)")
        bmi = st.number_input("Indeks Massa Tubuh (BMI)", min_value=0.0, value=25.0, help="Indeks Massa Tubuh (berat dalam kg/(tinggi dalam m)^2)")
        dpf = st.number_input("Fungsi Silsilah Diabetes (Diabetes Pedigree Function)", min_value=0.0, value=0.5, format="%.3f", help="Fungsi yang mengukur kemungkinan diabetes berdasarkan riwayat keluarga")
        age = st.number_input("Usia (Age)", min_value=0, value=30, help="Usia dalam tahun")

    submitted = st.form_submit_button("Prediksi Risiko")

if submitted:
    # Buat turunan fitur
    if insulin <= 80:
        insulin_level = "Low"
    elif insulin <= 120:
        insulin_level = "Normal"
    else:
        insulin_level = "Prediabet"

    if bmi < 18.5:
        weight_category = "Underweight"
    elif bmi < 25:
        weight_category = "Normal"
    elif bmi < 30:
        weight_category = "Overweight"
    else:
        weight_category = "Obese"

    # Susun dataframe input
    input_df = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
        "WeightCategory": weight_category,
        "InsulinLevel": insulin_level
    }])

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1] * 100

    # Tampilkan hasil
    st.subheader("Hasil Prediksi")
    if prediction == 1:
        st.error(f"**Hasil Prediksi: Positif Diabetes** (Probabilitas: **{prob:.2f}%**)")
        st.warning("Disarankan untuk segera berkonsultasi dengan dokter untuk diagnosis dan penanganan lebih lanjut.")
    else:
        st.success(f"**Hasil Prediksi: Negatif Diabetes** (Probabilitas: **{prob:.2f}%**)")
        st.info("Tetap jaga pola hidup sehat untuk mencegah diabetes.")

    # ---
    # Tambahkan bagian untuk Global Feature Importance (Koefisien Model) - Hanya 8 fitur asli
    st.markdown("---")
    st.subheader("📊 Kepentingan Fitur Global (Koefisien Model - 8 Fitur Asli):")

    try:
        clf = model.named_steps['classifier']

        if hasattr(clf, 'coef_'):
            coefficients = clf.coef_[0]

            # Mendapatkan indeks kolom numerik dari feature_names_transformed
            numerical_indices = [i for i, feature in enumerate(feature_names_transformed) if feature in original_numerical_columns]

            # Buat DataFrame hanya untuk fitur numerik
            filtered_numerical_importance_df = pd.DataFrame({
                'Feature': [feature_names_transformed[i] for i in numerical_indices],
                'Importance': [abs(coefficients[i]) for i in numerical_indices]
            })

            # Urutkan berdasarkan kepentingan dari yang terbesar
            filtered_numerical_importance_df = filtered_numerical_importance_df.sort_values(by='Importance', ascending=False)

            # Buat plot batang menggunakan seaborn
            fig_imp, ax_imp = plt.subplots(figsize=(10, 7))
            sns.barplot(x='Importance', y='Feature', data=filtered_numerical_importance_df, ax=ax_imp, palette='viridis')
            ax_imp.set_title("Kepentingan Fitur Global (8 Fitur Asli)")
            ax_imp.set_xlabel("Magnitudo Koefisien (Kepentingan)")
            ax_imp.set_ylabel("Fitur")
            plt.tight_layout()
            st.pyplot(fig_imp)
            plt.close(fig_imp)

            st.caption("Grafik ini menunjukkan seberapa penting masing-masing dari 8 fitur asli secara umum bagi model, diukur dari magnitudo koefisiennya.")

        else:
            st.info("Model yang digunakan tidak menyediakan 'coef_' untuk visualisasi kepentingan fitur global.")

    except Exception as e:
        st.warning(f"Tidak dapat menampilkan plot kepentingan fitur global: {e}")


st.markdown("---")
st.markdown("Aplikasi ini dibuat untuk tujuan edukasi dan tidak menggantikan nasihat medis profesional.")
