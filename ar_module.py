import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime
from fileupload import get_processed_df

def ar_module():
    st.title("💰 Accounts Receivable (AR) - Enhanced")
    
    # ---------------- Upload Invoice/Sales Files ---------------- #
    uploaded_files = st.file_uploader("Upload Invoice / Sales CSV/XLSX files", type=['csv','xlsx'], accept_multiple_files=True)
    
    ar_df = pd.DataFrame()
    for f in uploaded_files:
        try:
            df = pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f)
            ar_df = pd.concat([ar_df, df], ignore_index=True)
        except Exception as e:
            st.error(f"Failed to read {f.name}: {e}")
    
    if ar_df.empty:
        st.info("Upload invoice/sales files to proceed.")
        return

    # ---------------- Normalize ---------------- #
    ar_df.columns = [c.strip().title() for c in ar_df.columns]
    if 'Amount' not in ar_df.columns:
        ar_df['Amount'] = 0.0
    if 'Paid' not in ar_df.columns:
        ar_df['Paid'] = 0.0
    if 'Invoice Date' not in ar_df.columns:
        ar_df['Invoice Date'] = pd.Timestamp.today()
    else:
        ar_df['Invoice Date'] = pd.to_datetime(ar_df['Invoice Date'], errors='coerce')

    # ---------------- Outstanding / Received ---------------- #
    bank_df = get_processed_df()
    if not bank_df.empty:
        # Match payments: Description contains invoice/customer & Amount <= invoice
        def match_payment(row):
            matches = bank_df[
                (bank_df['Amount'] <= row['Amount']) &
                (bank_df['Description'].str.contains(str(row.get('Description','')), case=False))
            ]
            total_paid = matches['Amount'].sum() if not matches.empty else 0
            return total_paid
        ar_df['Received'] = ar_df.apply(match_payment, axis=1)
    else:
        ar_df['Received'] = 0.0

    ar_df['Outstanding'] = ar_df['Amount'] - ar_df['Received']
    ar_df['Paid Status'] = ar_df['Outstanding'].apply(lambda x: "Paid" if x <= 0 else "Partial" if x < ar_df['Amount'].max() else "Unpaid")

    # ---------------- Aging Buckets ---------------- #
    today = pd.Timestamp.today()
    ar_df['Days Outstanding'] = (today - ar_df['Invoice Date']).dt.days
    def aging_bucket(days):
        if days <= 30:
            return "0-30"
        elif days <= 60:
            return "31-60"
        elif days <= 90:
            return "61-90"
        else:
            return "90+"
    ar_df['Aging Bucket'] = ar_df['Days Outstanding'].apply(aging_bucket)

    # ---------------- Dashboard Metrics ---------------- #
    st.subheader("📊 AR Dashboard")
    st.metric("Total AR", f"{ar_df['Amount'].sum():,.2f}")
    st.metric("Received", f"{ar_df['Received'].sum():,.2f}")
    st.metric("Outstanding", f"{ar_df['Outstanding'].sum():,.2f}")
    st.metric("Overdue (>30 days)", f"{ar_df[ar_df['Days Outstanding']>30]['Outstanding'].sum():,.2f}")

    # ---------------- Filters ---------------- #
    st.subheader("🔍 Filter AR Transactions")
    customer_filter = st.text_input("Customer/Description Filter")
    status_filter = st.selectbox("Payment Status", ['All','Paid','Partial','Unpaid'])
    bucket_filter = st.selectbox("Aging Bucket", ['All','0-30','31-60','61-90','90+'])

    filtered_df = ar_df.copy()
    if customer_filter:
        filtered_df = filtered_df[filtered_df['Description'].str.contains(customer_filter, case=False)]
    if status_filter != 'All':
        filtered_df = filtered_df[filtered_df['Paid Status'] == status_filter]
    if bucket_filter != 'All':
        filtered_df = filtered_df[filtered_df['Aging Bucket'] == bucket_filter]

    st.dataframe(filtered_df)

    # ---------------- Downloads ---------------- #
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", data=csv_data, file_name="ar_report.csv")
    
    excel_output = BytesIO()
    with pd.ExcelWriter(excel_output, engine='xlsxwriter') as writer:
        filtered_df.to_excel(writer, index=False, sheet_name='AR')
    excel_output.seek(0)
    st.download_button("⬇️ Download Excel", data=excel_output, file_name="ar_report.xlsx")
