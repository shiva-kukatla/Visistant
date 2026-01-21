import streamlit as st
import pandas as pd
import plotly.express as px
import traceback
import re
import google.generativeai as genai

# --- PAGE CONFIG ---
st.set_page_config(page_title="Visistant", layout="wide")
st.title("🧠 Visistant - Natural Language to Data Visualization")

# --- GEMINI SETUP ---
genai.configure(api_key="AIzaSyAg0rd4rXa1J6x88CaXs6e4OqMOdLwVwuo")  # Replace with your key
model = genai.GenerativeModel("gemini-2.5-flash-lite")

# --- FILE UPLOAD ---
uploaded_file = st.sidebar.file_uploader("📤 Upload a CSV file", type=["csv"])

# --- CLEANING FUNCTION ---
def preprocess_dataframe(df):
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates()

    for col in df_clean.select_dtypes(include=['float64', 'int64']).columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

    for col in df_clean.select_dtypes(include=['object', 'category']).columns:
        if df_clean[col].isnull().any():
            mode = df_clean[col].mode()
            df_clean[col] = df_clean[col].fillna(mode[0] if not mode.empty else "Unknown")

    return df_clean

# --- MAIN APP LOGIC ---
if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    df_cleaned = preprocess_dataframe(df_raw)

    st.subheader("📄 Raw Data (First 5 Rows)")
    st.dataframe(df_raw.head())

    st.subheader("🧽 Cleaned Data (First 5 Rows)")
    st.dataframe(df_cleaned.head())

    with st.expander("🧾 Data Cleaning Summary"):
        st.write("✅ Duplicates removed")
        st.write("✅ NaN numeric values → filled with column mean")
        st.write("✅ NaN categorical values → filled with column mode")

    query = st.text_input("💬 Ask a question to visualize (e.g. 'Total sales by region'): ")

    if query:
        # Step 1: Build prompt with dataset info
        def build_prompt(error_message=None):
            col_info = "\n".join([f"- {col}: {dtype}" for col, dtype in zip(df_cleaned.columns, df_cleaned.dtypes)])
            prompt = f"""
You are a Python data analyst. A cleaned pandas DataFrame named `df` is loaded with these columns:
{col_info}

- Missing values are already imputed.
- Duplicates are removed.
- Use Plotly (not matplotlib).
- Use: `fig = px...` and DO NOT use `fig.show()`

User Query: "{query}"

Generate only the required Plotly Python code (inside ```python ... ``` markdown). Add chart title.
"""
            if error_message:
                prompt += f"\n\nNote: The previous code failed with error:\n{error_message}\nPlease fix it and regenerate correct code."
            return prompt

        attempt = 0
        success = False
        code_block = ""

        while attempt < 3 and not success:
            try:
                response = model.generate_content(build_prompt(error_message=None if attempt == 0 else error_text))
                code_block = response.text

                # Extract Python code from markdown
                match = re.search(r"```(?:python)?\n(.*?)```", code_block, re.DOTALL)
                code = match.group(1).strip() if match else code_block

                # Execute safely
                local_vars = {'df': df_cleaned}
                exec(code, {}, local_vars)
                fig = local_vars.get('fig', None)

                if fig:
                    st.success(f"✅ Plot generated (attempt {attempt+1})")
                    st.code(code, language='python')
                    st.plotly_chart(fig, use_container_width=True)
                    success = True
                else:
                    raise Exception("Plotly figure `fig` not returned in code.")

            except Exception as e:
                error_text = traceback.format_exc()
                st.warning(f"⚠️ Attempt {attempt+1} failed:\n{e}")
                attempt += 1

        if not success:
            st.error("❌ Could not generate a valid plot after 3 attempts.")
            st.code(code_block, language='python')

