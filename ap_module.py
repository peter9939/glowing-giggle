import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime
from fileupload import get_processed_df

def ap_module():
    st.title("📦 Accounts Payable (AP) - Enhanced")
    
    # ---------------- Upload Vendor/Bill Files ---------------- #
    uploaded_files = st.file_uploader("Upload Vendor/Bill CSV/XLSX files", type=['csv','xlsx'], accept_multiple_files=True)
    
    ap_df = pd.DataFrame()
    for f in uploaded_files:
        try:
            df = pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f)
            ap_df = pd.concat([ap_df, df], ignore_index=True)
        except Exception as e:
            st.error(f"Failed to read {f.name}: {e}")
    
    if ap_df.empty:
        st.info("Upload bill/vendor files to proceed.")
        return

    # ---------------- Normalize ---------------- #
    ap_df.columns = [c.strip().title() for c in ap_df.columns]
    if 'Amount' not in ap_df.columns:
        ap_df['Amount'] = 0.0
    if 'Paid' not in ap_df.columns:
        ap_df['Paid'] = 0.0
    if 'Bill Date' not in ap_df.columns:
        ap_df['Bill Date'] = pd.Timestamp.today()
    else:
        ap_df['Bill Date'] = pd.to_datetime(ap_df['Bill Date'], errors='coerce')

    # ---------------- Outstanding / Paid ---------------- #
    bank_df = get_processed_df()
    if not bank_df.empty:
        def match_payment(row):
            matches = bank_df[
                (bank_df['Amount'] >= -row['Amount']) &
                (bank_df['Description'].str.contains(str(row.get('Description','')), case=False))
            ]
            total_paid = -matches['Amount'].sum() if not matches.empty else 0
            return total_paid
        ap_df['Paid'] = ap_df.apply(match_payment, axis=1)
    else:
        ap_df['Paid'] = 0.0

    ap_df['Outstanding'] = ap_df['Amount'] - ap_df['Paid']
    ap_df['Payment Status'] = ap_df['Outstanding'].apply(lambda x: "Paid" if x <= 0 else "Partial" if x < ap_df['Amount'].max() else "Unpaid")

    # ---------------- Aging Buckets ---------------- #
    today = pd.Timestamp.today()
    ap_df['Days Outstanding'] = (today - ap_df['Bill Date']).dt.days
    def aging_bucket(days):
        if days <= 30:
            return "0-30"
        elif days <= 60:
            return "31-60"
        elif days <= 90:
            return "61-90"
        else:
            return "90+"
    ap_df['Aging Bucket'] = ap_df['Days Outstanding'].apply(aging_bucket)

    # ---------------- Dashboard Metrics ---------------- #
    st.subheader("📊 AP Dashboard")
    st.metric("Total AP", f"{ap_df['Amount'].sum():,.2f}")
    st.metric("Paid", f"{ap_df['Paid'].sum():,.2f}")
    st.metric("Outstanding", f"{ap_df['Outstanding'].sum():,.2f}")
    st.metric("Overdue (>30 days)", f"{ap_df[ap_df['Days Outstanding']>30]['Outstanding'].sum():,.2f}")

    # ---------------- Filters ---------------- #
    st.subheader("🔍 Filter AP Transactions")
    vendor_filter = st.text_input("Vendor/Description Filter")
    status_filter = st.selectbox("Payment Status", ['All','Paid','Partial','Unpaid'])
    bucket_filter = st.selectbox("Aging Bucket", ['All','0-30','31-60','61-90','90+'])

    filtered_df = ap_df.copy()
    if vendor_filter:
        filtered_df = filtered_df[filtered_df['Description'].str.contains(vendor_filter, case=False)]
    if status_filter != 'All':
        filtered_df = filtered_df[filtered_df['Payment Status'] == status_filter]
    if bucket_filter != 'All':
        filtered_df = filtered_df[filtered_df['Aging Bucket'] == bucket_filter]

    st.dataframe(filtered_df)

    # ---------------- Downloads ---------------- #
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", data=csv_data, file_name="ap_report.csv")
    
    excel_output = BytesIO()
    with pd.ExcelWriter(excel_output, engine='xlsxwriter') as writer:
        filtered_df.to_excel(writer, index=False, sheet_name='AP')
    excel_output.seek(0)
    st.download_button("⬇️ Download Excel", data=excel_output, file_name="ap_report.xlsx")
