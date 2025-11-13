import streamlit as st
import tempfile
import os
import sys
from dotenv import load_dotenv  # pip install python-dotenv

# Load .env file (for local testing only)
load_dotenv()

# Add src/ to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from orchestrator import Orchestrator

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Agentic DataLab", page_icon="🤖")
st.title("Agentic DataLab — AI-powered Data Science Automation")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
target = st.text_input("Target column name")

# Function to display errors
def show_error(msg):
    st.error(f"❌ Error: {msg}")

if uploaded and target:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded.getvalue())
            tmp.flush()

            orch = Orchestrator()

            with st.spinner("Running pipeline... This may take a few moments"):
                result = orch.run_pipeline(tmp.name, target)

            st.success("✅ Pipeline completed successfully!")
            st.write("### 🧠 Best Model:", result.get('best_model', 'N/A'))
            st.write("### 📊 Model Scores")
            st.json(result.get('scores', {}))
            st.write("### 📑 Evaluation Report")
            st.write(result.get('report', 'No report generated.'))

            os.unlink(tmp.name)

    except FileNotFoundError as fe:
        show_error("Temporary file not found. Please try again.")
    except ValueError as ve:
        show_error(f"ValueError: {ve}")
    except Exception as e:
        show_error(f"Unexpected error: {e}")

elif uploaded and not target:
    st.warning("⚠️ Please enter the target column name before running the pipeline.")

import os
from dotenv import load_dotenv

# Load environment variables from .env (for local testing)
load_dotenv()


