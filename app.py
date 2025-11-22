import streamlit as st

# ---------------- Import Modules ---------------- #
from fileupload import file_upload_module
from dashboard import dashboard_module
from audit import audit_module
from tax import tax_module
from budget import budget_module
from reconciliation import reconciliation_module

# ---------------- App Setup ---------------- #
st.set_page_config(page_title="📊 Professional Accounting Suite", layout="wide")
st.title("📊 Professional Accounting Suite")

# ---------------- Sidebar Navigation ---------------- #
st.sidebar.title("Navigation")
menu_options = [
    "1️⃣ Upload Transactions",
    "2️⃣ Dashboard",
    "3️⃣ Audit & Data Validation",
    "4️⃣ Tax Management",
    "5️⃣ Budgeting & Forecasting",
    "6️⃣ Bank Reconciliation"
]
choice = st.sidebar.radio("Go to", menu_options)

# ---------------- Module Routing ---------------- #
if choice == "1️⃣ Upload Transactions":
    st.header("1️⃣ Upload Transactions")
    file_upload_module()

elif choice == "2️⃣ Dashboard":
    st.header("2️⃣ Dashboard")
    dashboard_module()

elif choice == "3️⃣ Audit & Data Validation":
    st.header("3️⃣ Audit & Data Validation")
    audit_module()

elif choice == "4️⃣ Tax Management":
    st.header("4️⃣ Tax Management")
    tax_module()

elif choice == "5️⃣ Budgeting & Forecasting":
    st.header("5️⃣ Budgeting & Forecasting")
    budget_module()

elif choice == "6️⃣ Bank Reconciliation":
    st.header("6️⃣ Bank Reconciliation")
    reconciliation_module()


# ---------------- Footer ---------------- #
st.markdown("---")
st.markdown("© 2025 Professional Accounting Suite. All rights reserved.")
