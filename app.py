import streamlit as st

# ---------------- Import Modules ---------------- #
from fileupload import file_upload_module
from dashboard import dashboard_module
from audit import audit_module
from tax import tax_module
from budget import budget_module
from reconciliation import reconciliation_module
from ar_module import ar_module
from ap_module import ap_module

menu_options = [
    "1️⃣ Upload Transactions",
    "2️⃣ Dashboard",
    "3️⃣ Audit & Data Validation",
    "4️⃣ Tax Management",
    "5️⃣ Budgeting & Forecasting",
    "6️⃣ Bank Reconciliation",
    "7️⃣ Accounts Receivable",
    "8️⃣ Accounts Payable"
]

choice = st.sidebar.radio("Go to", menu_options)

if choice == "1️⃣ Upload Transactions":
    file_upload_module()
elif choice == "2️⃣ Dashboard":
    dashboard_module()
elif choice == "3️⃣ Audit & Data Validation":
    audit_module()
elif choice == "4️⃣ Tax Management":
    tax_module()
elif choice == "5️⃣ Budgeting & Forecasting":
    budget_module()
elif choice == "6️⃣ Bank Reconciliation":
    reconciliation_module()
elif choice == "7️⃣ Accounts Receivable":
    ar_module()
elif choice == "8️⃣ Accounts Payable":
    ap_module()
