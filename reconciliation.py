import streamlit as st
import pandas as pd
from rapidfuzz import fuzz
from io import BytesIO

# ---------------- Reconciliation Function ---------------- #
def reconcile(bank_df, ledger_df, desc_threshold=80, date_tolerance=2):
    bank_df = bank_df.copy()
    ledger_df = ledger_df.copy()
    
    # Ensure required columns
    for col in ['Date', 'Description', 'Amount']:
        if col not in ledger_df.columns:
            ledger_df[col] = pd.NaT if col == 'Date' else 'Unknown' if col == 'Description' else 0
        if col not in bank_df.columns:
            bank_df[col] = pd.NaT if col == 'Date' else 'Unknown' if col == 'Description' else 0

    # Convert to datetime safely
    ledger_df['Date'] = pd.to_datetime(ledger_df['Date'], errors='coerce')
    bank_df['Date'] = pd.to_datetime(bank_df['Date'], errors='coerce')

    # Drop rows with invalid dates
    ledger_df = ledger_df.dropna(subset=['Date'])
    bank_df = bank_df.dropna(subset=['Date'])

    ledger_df['Reconciled'] = False
    bank_df['Matched_Ledger_Index'] = -1

    # Match bank transactions to ledger
    for b_idx, b_row in bank_df.iterrows():
        best_match_idx = None
        best_score = 0
        for l_idx, l_row in ledger_df.iterrows():
            # Amount must match (tolerance 0.01)
            if abs(b_row['Amount'] - l_row['Amount']) > 0.01:
                continue
            # Description fuzzy match
            score = fuzz.token_sort_ratio(str(b_row['Description']), str(l_row['Description']))
            # Date difference in days
            date_diff = abs((b_row['Date'] - l_row['Date']).days)
            if score >= desc_threshold and date_diff <= date_tolerance and score > best_score:
                best_score = score
                best_match_idx = l_idx
        if best_match_idx is not None:
            ledger_df.at[best_match_idx, 'Reconciled'] = True
            bank_df.at[b_idx, 'Matched_Ledger_Index'] = best_match_idx

    # Split results
    matched = ledger_df[ledger_df['Reconciled'] == True]
    unmatched_ledger = ledger_df[ledger_df['Reconciled'] == False]
    unmatched_bank = bank_df[bank_df['Matched_Ledger_Index'] == -1]

    return matched, unmatched_ledger, unmatched_bank

# ---------------- Export Function ---------------- #
def to_excel(df_dict):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in df_dict.items():
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()

# ---------------- Streamlit App ---------------- #
def reconciliation_module():
    st.title("🔄 Bank Reconciliation")

    st.info("Upload both Ledger and Bank Statement files (CSV/XLSX).")

    ledger_file = st.file_uploader("Upload Ledger File", type=['csv','xlsx'], key="ledger")
    bank_file = st.file_uploader("Upload Bank Statement", type=['csv','xlsx'], key="bank")

    desc_threshold = st.sidebar.slider("Description Match Threshold (%)", 50, 100, 80)
    date_tolerance = st.sidebar.number_input("Date Tolerance (days)", min_value=0, max_value=10, value=2, step=1)

    if ledger_file and bank_file:
        # Load files
        try:
            ledger_df = pd.read_csv(ledger_file) if ledger_file.name.endswith('.csv') else pd.read_excel(ledger_file)
            bank_df = pd.read_csv(bank_file) if bank_file.name.endswith('.csv') else pd.read_excel(bank_file)
        except Exception as e:
            st.error(f"Failed to read files: {e}")
            return

        # Run reconciliation
        matched, unmatched_ledger, unmatched_bank = reconcile(bank_df, ledger_df, desc_threshold, date_tolerance)

        st.subheader("✅ Matched Transactions")
        st.dataframe(matched)

        st.subheader("⚠️ Unmatched Ledger Transactions")
        st.dataframe(unmatched_ledger)

        st.subheader("⚠️ Unmatched Bank Transactions")
        st.dataframe(unmatched_bank)

        # Download Excel
        df_dict = {
            'Matched': matched,
            'Unmatched_Ledger': unmatched_ledger,
            'Unmatched_Bank': unmatched_bank
        }
        st.download_button("⬇️ Download Reconciliation Report", data=to_excel(df_dict), file_name="reconciliation_report.xlsx")

if __name__ == "__main__":
    reconciliation_module()
