import streamlit as st
import pandas as pd
from io import BytesIO

# ---------------- Load Processed Data ---------------- #
def load_data():
    if 'processed_df' not in st.session_state:
        st.warning("No processed data found. Please upload your transactions first.")
        return pd.DataFrame()

    df = st.session_state['processed_df'].copy()

    # Ensure 'Date' column exists
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').fillna(pd.Timestamp.today())
    else:
        df['Date'] = pd.Series([pd.Timestamp.today()] * len(df))

    # Ensure necessary columns
    for col in ['Account','Main Category','Subcategory','Balance Change','Debit','Credit','Description','Auto-Matched']:
        if col not in df.columns:
            if col in ['Balance Change','Debit','Credit']:
                df[col] = 0
            elif col == 'Auto-Matched':
                df[col] = True
            else:
                df[col] = 'Unknown'

    # Ensure numeric columns
    df['Balance Change'] = pd.to_numeric(df['Balance Change'], errors='coerce').fillna(0)
    df['Debit'] = pd.to_numeric(df['Debit'], errors='coerce').fillna(0)
    df['Credit'] = pd.to_numeric(df['Credit'], errors='coerce').fillna(0)

    return df

# ---------------- Outlier Detection ---------------- #
def detect_outliers(df, column='Balance Change', method='zscore', threshold=3):
    if df.empty or column not in df.columns:
        return pd.Series([False]*len(df), index=df.index)
    try:
        if method == 'zscore':
            mean = df[column].mean()
            std = df[column].std()
            if std == 0:
                return pd.Series([False]*len(df), index=df.index)
            return abs(df[column] - mean) / std > threshold
        elif method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:
                return pd.Series([False]*len(df), index=df.index)
            return (df[column] < Q1 - 1.5*IQR) | (df[column] > Q3 + 1.5*IQR)
    except Exception:
        return pd.Series([False]*len(df), index=df.index)

# ---------------- Category Mismatch Detection ---------------- #
def detect_category_mismatch(df):
    if 'Auto-Matched' not in df.columns:
        return pd.Series([False]*len(df), index=df.index)
    try:
        return df['Auto-Matched'] == False
    except Exception:
        return pd.Series([False]*len(df), index=df.index)

# ---------------- Export to Excel ---------------- #
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Audit Report')
    return output.getvalue()

# ---------------- Highlight Function ---------------- #
def highlight_flags(row, outliers_idx, mismatches_idx):
    if row.name in outliers_idx:
        return ['background-color: #ffcccc']*len(row)
    elif row.name in mismatches_idx:
        return ['background-color: #fff3cd']*len(row)
    else:
        return ['']*len(row)

# ---------------- Audit Module ---------------- #
def audit_module():
    st.title("🕵️‍♂️ Audit & Data Validation Dashboard")

    df = load_data()
    if df.empty:
        return

    # ---------------- Sidebar Filters ---------------- #
    st.sidebar.header("Filters")
    start_date = st.sidebar.date_input("Start Date", df['Date'].min())
    end_date = st.sidebar.date_input("End Date", df['Date'].max())
    main_categories = st.sidebar.multiselect("Main Category", df['Main Category'].unique())
    sub_categories = st.sidebar.multiselect("Subcategory", df['Subcategory'].unique())
    outlier_method = st.sidebar.selectbox("Outlier Detection Method", ['zscore', 'iqr'])
    outlier_threshold = st.sidebar.number_input("Outlier Threshold", min_value=0.1, value=3.0, step=0.1)

    # ---------------- Filter Data ---------------- #
    df_filtered = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
    if main_categories:
        df_filtered = df_filtered[df_filtered['Main Category'].isin(main_categories)]
    if sub_categories:
        df_filtered = df_filtered[df_filtered['Subcategory'].isin(sub_categories)]

    if df_filtered.empty:
        st.info("No transactions found for selected filters.")
        return

    # ---------------- Detect Flags ---------------- #
    outliers_idx = df_filtered[detect_outliers(df_filtered, method=outlier_method, threshold=outlier_threshold)].index
    mismatches_idx = df_filtered[detect_category_mismatch(df_filtered)].index

    # ---------------- KPIs ---------------- #
    st.subheader("📊 Key Metrics")
    total_txn = len(df_filtered)
    total_outliers = len(outliers_idx)
    total_mismatches = len(mismatches_idx)
    cols = st.columns(3)
    cols[0].metric("Total Transactions", total_txn)
    cols[1].metric("Total Outliers", total_outliers)
    cols[2].metric("Total Category Mismatches", total_mismatches)

    # ---------------- Display Flagged Data ---------------- #
    st.subheader("🚨 Flagged Transactions")
    flagged = pd.concat([df_filtered.loc[outliers_idx], df_filtered.loc[mismatches_idx]]).drop_duplicates()
    if not flagged.empty:
        st.dataframe(flagged.style.apply(highlight_flags, outliers_idx=outliers_idx, mismatches_idx=mismatches_idx, axis=1))
        st.download_button("⬇️ Download Flagged Transactions (Excel)", data=to_excel(flagged), file_name="audit_flagged.xlsx")
    else:
        st.info("No flagged transactions detected.")

if __name__ == "__main__":
    audit_module()
