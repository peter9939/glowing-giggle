# ar_module.py
import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import timedelta
from rapidfuzz import process, fuzz
from fileupload import get_processed_df

# Tunable thresholds
FUZZY_NAME_THRESHOLD = 75
DATE_TOLERANCE_DAYS = 7
AMOUNT_TOLERANCE = 0.01

def normalize_text(s):
    return "" if pd.isna(s) else str(s).strip().lower()

def load_internal_file_flexible(uploaded_file):
    """Load internal master file and standardize columns (flexible header names)."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Failed to load internal file: {e}")
        return pd.DataFrame()

    # normalize column names
    cols_map = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    df.rename(columns={orig: cols_map[orig] for orig in df.columns}, inplace=True)

    # helper to find likely column name among variants
    def find_col(variants):
        for v in variants:
            if v in df.columns:
                return v
        return None

    type_col = find_col(['type','record_type','ar_ap','kind'])
    id_col = find_col(['invoice_id','invoice_number','invoice','id','document_id','ref'])
    name_col = find_col(['customer','vendor','name','payee','party'])
    amount_col = find_col(['amount','amt','value','total'])
    date_col = find_col(['date','invoice_date','issue_date','bill_date'])
    due_col = find_col(['due_date','duedate'])

    # fill defaults
    if type_col is None:
        df['type'] = ''
        type_col = 'type'
    if id_col is None:
        df['invoice_id'] = ''
        id_col = 'invoice_id'
    if name_col is None:
        df['name'] = ''
        name_col = 'name'
    if amount_col is None:
        df['amount'] = 0.0
        amount_col = 'amount'
    if date_col is None:
        df['date'] = pd.NaT
        date_col = 'date'
    if due_col is None:
        df['due_date'] = pd.NaT
        due_col = 'due_date'

    # standardize types & types content
    df[type_col] = df[type_col].astype(str).fillna('').str.upper()
    df[id_col] = df[id_col].astype(str).fillna('')
    df[name_col] = df[name_col].astype(str).fillna('')
    df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce').fillna(0.0)
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[due_col] = pd.to_datetime(df[due_col], errors='coerce')

    # infer type when blank: positive -> AR, negative -> AP (helpful but not perfect)
    def infer_type(row):
        t = str(row[type_col]).strip().lower()
        if t in ('ar','invoice','receivable','sale','sales','customer'):
            return 'AR'
        if t in ('ap','bill','payable','purchase','vendor','expense'):
            return 'AP'
        # fallback by sign: positive -> AR (incoming), negative -> AP (outgoing)
        return 'AR' if row[amount_col] >= 0 else 'AP'

    df['record_type'] = df.apply(infer_type, axis=1)

    # produce canonical standardized DF
    std = pd.DataFrame({
        'type': df['record_type'],
        'invoice_id': df[id_col].astype(str),
        'name': df[name_col].astype(str),
        # For AR we expect positive amounts (incoming). Keep sign as-is.
        'amount': df[amount_col].astype(float),
        'date': df[date_col],
        'due_date': df[due_col]
    })

    return std

def build_bank_index(bank_df):
    """Index bank processed df (from fileupload.get_processed_df())."""
    b = bank_df.copy()
    # try several name conventions
    if 'date' not in b.columns and 'Date' in b.columns:
        b['date'] = pd.to_datetime(b['Date'], errors='coerce')
    else:
        b['date'] = pd.to_datetime(b.get('date', pd.NaT), errors='coerce')
    if 'description' not in b.columns and 'Description' in b.columns:
        b['description'] = b['Description'].astype(str)
    else:
        b['description'] = b.get('description', '').astype(str)
    if 'amount' not in b.columns and 'Amount' in b.columns:
        b['amount'] = pd.to_numeric(b['Amount'], errors='coerce').fillna(0.0)
    else:
        b['amount'] = pd.to_numeric(b.get('amount', 0.0), errors='coerce').fillna(0.0)

    b['desc_norm'] = b['description'].str.strip().str.lower()
    b = b.reset_index().rename(columns={'index': 'bank_index'})
    b['matched'] = False
    b['matched_refs'] = [[] for _ in range(len(b))]
    return b

def match_ar_to_bank(ar_row, bank_idx):
    """
    Attempt to match AR invoice to bank transactions (incoming payments).
    Returns list of bank_index matched and total matched amount.
    """
    invoice_amt = round(float(ar_row['amount']), 2)
    if invoice_amt <= 0:
        # skip non-positive AR amounts
        return [], 0.0

    candidates = bank_idx[~bank_idx['matched']].copy()

    # narrow by date tolerance if invoice date available
    if pd.notna(ar_row['date']):
        min_d = ar_row['date'] - timedelta(days=DATE_TOLERANCE_DAYS)
        max_d = ar_row['date'] + timedelta(days=DATE_TOLERANCE_DAYS)
        candidates = candidates[(candidates['date'] >= min_d) & (candidates['date'] <= max_d)]

    # 1) If invoice_id appears in bank description -> exact
    invoice_id = normalize_text(ar_row['invoice_id'])
    if invoice_id:
        exact = candidates[candidates['desc_norm'].str.contains(invoice_id, na=False)]
        if not exact.empty:
            for _, r in exact.iterrows():
                if abs(r['amount'] - invoice_amt) <= AMOUNT_TOLERANCE:
                    return [int(r['bank_index'])], r['amount']
            # allow summing exacts for partials (rare) - greedy
            total = 0.0; idxs = []
            for _, r in exact.iterrows():
                if r['amount'] > 0:
                    idxs.append(int(r['bank_index'])); total += r['amount']
                    if abs(total - invoice_amt) <= AMOUNT_TOLERANCE:
                        return idxs, total

    # 2) Fuzzy name matching + amount exact
    name = normalize_text(ar_row['name'])
    if name and not candidates.empty:
        choices = candidates['desc_norm'].tolist()
        res = process.extract(name, choices, scorer=fuzz.token_sort_ratio, limit=20)
        # Attempt to find candidate whose amount equals invoice_amt
        for choice, score, idx in res:
            if score < FUZZY_NAME_THRESHOLD:
                continue
            candidate = candidates.iloc[idx]
            if candidate['amount'] > 0 and abs(candidate['amount'] - invoice_amt) <= AMOUNT_TOLERANCE:
                return [int(candidate['bank_index'])], candidate['amount']
        # If not single, try to sum positive candidates with good fuzzy score
        matched_idxs = []; total = 0.0
        for choice, score, idx in res:
            if score < FUZZY_NAME_THRESHOLD:
                continue
            candidate = candidates.iloc[idx]
            if candidate['amount'] > 0:
                matched_idxs.append(int(candidate['bank_index'])); total += candidate['amount']
                if abs(total - invoice_amt) <= AMOUNT_TOLERANCE:
                    return matched_idxs, total

    # 3) Fallback: amount-only match (within date tolerance)
    amt_candidates = candidates[abs(candidates['amount'] - invoice_amt) <= AMOUNT_TOLERANCE]
    if not amt_candidates.empty:
        r = amt_candidates.iloc[0]
        if r['amount'] > 0:
            return [int(r['bank_index'])], r['amount']

    return [], 0.0

def ar_module():
    st.title("💰 Accounts Receivable (AR) - Production Ready")
    st.info("Upload internal master file (may contain both AR/AP). Bank transactions must be uploaded first in 'Upload Transactions' to session.")

    uploaded_file = st.file_uploader("Internal Master File (CSV/XLSX)", type=['csv','xlsx'], key='ar_internal')
    if uploaded_file is None:
        st.info("Please upload the internal master file containing invoices (AR).")
        return

    internal = load_internal_file_flexible(uploaded_file)
    if internal.empty:
        st.error("Could not process internal file.")
        return

    ar_df = internal[internal['type'] == 'AR'].copy()
    if ar_df.empty:
        st.warning("No AR rows found.")
        return

    bank_df = get_processed_df()
    if bank_df is None or bank_df.empty:
        st.warning("No bank transactions in session. Upload bank data under 'Upload Transactions'.")
        return

    bank_idx = build_bank_index(bank_df)

    # Prepare results
    ar_df['received_amount'] = 0.0
    ar_df['matched_bank_refs'] = [[] for _ in range(len(ar_df))]
    ar_df['payment_status'] = 'Unreceived'

    for i, row in ar_df.iterrows():
        matched_idxs, total = match_ar_to_bank(row, bank_idx)
        if matched_idxs:
            ar_df.at[i, 'received_amount'] = round(total,2)
            ar_df.at[i, 'matched_bank_refs'] = matched_idxs
            # mark bank rows matched
            for bi in matched_idxs:
                bank_idx.loc[bank_idx['bank_index'] == bi, 'matched'] = True
                bank_idx.loc[bank_idx['bank_index'] == bi, 'matched_refs'] = bank_idx.loc[bank_idx['bank_index'] == bi, 'matched_refs'].apply(lambda lst: lst + [row['invoice_id']])
            if abs(total - row['amount']) <= AMOUNT_TOLERANCE:
                ar_df.at[i, 'payment_status'] = 'Received'
            else:
                ar_df.at[i, 'payment_status'] = 'Partial'
        else:
            ar_df.at[i, 'payment_status'] = 'Unreceived'

    ar_df['outstanding'] = (ar_df['amount'] - ar_df['received_amount']).round(2)
    today = pd.Timestamp.today()
    ar_df['days_outstanding'] = (today - ar_df['date']).dt.days.fillna(0).astype(int)
    def aging(days):
        if days <= 30: return '0-30'
        if days <= 60: return '31-60'
        if days <= 90: return '61-90'
        return '90+'
    ar_df['aging_bucket'] = ar_df['days_outstanding'].apply(aging)

    # Metrics and filters
    st.subheader("AR Summary")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total AR", f"{ar_df['amount'].sum():,.2f}")
    c2.metric("Received", f"{ar_df['received_amount'].sum():,.2f}")
    c3.metric("Outstanding", f"{ar_df['outstanding'].sum():,.2f}")
    c4.metric("Overdue (>30d)", f"{ar_df[ar_df['days_outstanding']>30]['outstanding'].sum():,.2f}")

    name_filter = st.text_input("Customer filter")
    status_filter = st.selectbox("Payment Status", ['All','Received','Partial','Unreceived'])
    bucket_filter = st.selectbox("Aging Bucket", ['All','0-30','31-60','61-90','90+'])

    filtered = ar_df.copy()
    if name_filter:
        filtered = filtered[filtered['name'].str.contains(name_filter, case=False, na=False)]
    if status_filter != 'All':
        filtered = filtered[filtered['payment_status'] == status_filter]
    if bucket_filter != 'All':
        filtered = filtered[filtered['aging_bucket'] == bucket_filter]

    st.dataframe(filtered[['invoice_id','name','amount','received_amount','outstanding','payment_status','days_outstanding','aging_bucket','matched_bank_refs']])

    # Downloads
    st.download_button("⬇️ Download AR CSV", data=filtered.to_csv(index=False).encode('utf-8'), file_name='ar_report.csv')
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        filtered.to_excel(writer, index=False, sheet_name='AR')
        bank_unmatched = bank_idx[~bank_idx['matched']].copy()
        bank_unmatched.to_excel(writer, index=False, sheet_name='Unmatched_Bank')
    buf.seek(0)
    st.download_button("⬇️ Download AR Excel", data=buf, file_name='ar_report.xlsx')

    st.subheader("Unmatched Bank Transactions (possible receipts)")
    st.dataframe(bank_idx[~bank_idx['matched']][['bank_index','date','description','amount']])

    st.session_state['ar_results'] = ar_df
