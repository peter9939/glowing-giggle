# reconciliation_module.py
import streamlit as st
import pandas as pd
from io import BytesIO
from rapidfuzz import fuzz
from datetime import timedelta
from rapidfuzz import process, fuzz


# This reconciliation module is robust to unknown ledger structure.
# It will try to infer AR/AP rows and run matching; also provides filters.

def load_file_flexible(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        return pd.DataFrame()
    cols_map = {c: c.strip().lower().replace(' ', '_') for c in df.columns}
    df.rename(columns={orig: cols_map[orig] for orig in df.columns}, inplace=True)
    # Try to standardize main columns
    def find_col(list_opts):
        for o in list_opts:
            if o in df.columns:
                return o
        return None
    type_col = find_col(['type','record_type','ar_ap','kind'])
    id_col = find_col(['invoice_id','invoice_number','invoice','id','document_id','ref'])
    name_col = find_col(['name','customer','vendor','payee','party'])
    amount_col = find_col(['amount','amt','value','total'])
    date_col = find_col(['date','transaction_date','invoice_date','bill_date'])
    # set defaults
    if type_col is None:
        df['type'] = ''
        type_col='type'
    if id_col is None:
        df['invoice_id']=''
        id_col='invoice_id'
    if name_col is None:
        df['name']=''
        name_col='name'
    if amount_col is None:
        df['amount']=0.0
        amount_col='amount'
    if date_col is None:
        df['date']=pd.NaT
        date_col='date'
    # normalize
    df[type_col] = df[type_col].astype(str).fillna('').str.upper()
    df[id_col] = df[id_col].astype(str).fillna('')
    df[name_col] = df[name_col].astype(str).fillna('')
    df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce').fillna(0.0)
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    # infer record type similar to AP/AR modules
    def infer_type(row):
        t = str(row[type_col]).strip().lower()
        if t in ('ap','bill','payable','vendor','purchase','expense'): return 'AP'
        if t in ('ar','invoice','receivable','sale','customer'): return 'AR'
        return 'AR' if row[amount_col] >= 0 else 'AP'
    df['record_type'] = df.apply(infer_type, axis=1)
    # standardized df
    std = pd.DataFrame({
        'type': df['record_type'],
        'invoice_id': df[id_col].astype(str),
        'name': df[name_col].astype(str),
        'amount': df[amount_col].astype(float),
        'date': df[date_col]
    })
    return std

def build_bank_index(bank_df):
    b = bank_df.copy()
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
    b = b.reset_index().rename(columns={'index':'bank_index'})
    b['matched'] = False
    b['matched_refs'] = [[] for _ in range(len(b))]
    return b

def reconcile_combined(ledger_df, bank_df, desc_threshold=80, date_tolerance=7, fuzzy_threshold=75, amount_tol=0.01):
    ledger = ledger_df.copy()
    bank = bank_df.copy()
    ledger['matched_amount'] = 0.0
    ledger['status'] = 'Unmatched'
    bank_idx = build_bank_index(bank)
    # For each ledger row (prefer AR/amount>0 or AP with abs)
    for li, lrow in ledger.iterrows():
        inv_amt = abs(float(lrow['amount']))
        # prepare candidates
        # candidate selection depends on type: AR expects bank incoming (+), AP expects bank outgoing (-)
        candidates = bank_idx[~bank_idx['matched']].copy()
        # date filter
        if pd.notna(lrow['date']):
            min_d = lrow['date'] - timedelta(days=date_tolerance)
            max_d = lrow['date'] + timedelta(days=date_tolerance)
            candidates = candidates[(candidates['date'] >= min_d) & (candidates['date'] <= max_d)]
        # invoice id check
        inv_id = str(lrow.get('invoice_id','')).strip().lower()
        exact_matches = []
        if inv_id:
            exact_matches = candidates[candidates['desc_norm'].str.contains(inv_id, na=False)]
            if not exact_matches.empty:
                # choose matching sign & amount
                for _, cand in exact_matches.iterrows():
                    # check sign vs type
                    if lrow['type']=='AR' and cand['amount']>0 and abs(cand['amount']-inv_amt) <= amount_tol:
                        ledger.at[li,'matched_amount'] = cand['amount']
                        ledger.at[li,'status'] = 'Matched'
                        bank_idx.loc[bank_idx['bank_index']==cand['bank_index'],'matched']=True
                        bank_idx.loc[bank_idx['bank_index']==cand['bank_index'],'matched_refs'] = bank_idx.loc[bank_idx['bank_index']==cand['bank_index'],'matched_refs'].apply(lambda lst: lst+[lrow.get('invoice_id')])
                        break
                    if lrow['type']=='AP' and cand['amount']<0 and abs(abs(cand['amount'])-inv_amt) <= amount_tol:
                        ledger.at[li,'matched_amount'] = abs(cand['amount'])
                        ledger.at[li,'status'] = 'Matched'
                        bank_idx.loc[bank_idx['bank_index']==cand['bank_index'],'matched']=True
                        bank_idx.loc[bank_idx['bank_index']==cand['bank_index'],'matched_refs'] = bank_idx.loc[bank_idx['bank_index']==cand['bank_index'],'matched_refs'].apply(lambda lst: lst+[lrow.get('invoice_id')])
                        break
            if ledger.at[li,'status']=='Matched':
                continue
        # fuzzy name + amount attempt
        name = str(lrow.get('name','')).strip().lower()
        if name and not candidates.empty:
            choices = candidates['desc_norm'].tolist()
            res = process.extract(name, choices, scorer=fuzz.token_sort_ratio, limit=20)
            # try exact amount match with good fuzzy score
            for choice, score, idx in res:
                if score < fuzzy_threshold:
                    continue
                cand = candidates.iloc[idx]
                if lrow['type']=='AR' and cand['amount']>0 and abs(cand['amount']-inv_amt) <= amount_tol:
                    ledger.at[li,'matched_amount']=cand['amount']; ledger.at[li,'status']='Matched'
                    bank_idx.loc[bank_idx['bank_index']==cand['bank_index'],'matched']=True
                    bank_idx.loc[bank_idx['bank_index']==cand['bank_index'],'matched_refs'] = bank_idx.loc[bank_idx['bank_index']==cand['bank_index'],'matched_refs'].apply(lambda lst: lst+[lrow.get('invoice_id')])
                    break
                if lrow['type']=='AP' and cand['amount']<0 and abs(abs(cand['amount'])-inv_amt) <= amount_tol:
                    ledger.at[li,'matched_amount']=abs(cand['amount']); ledger.at[li,'status']='Matched'
                    bank_idx.loc[bank_idx['bank_index']==cand['bank_index'],'matched']=True
                    bank_idx.loc[bank_idx['bank_index']==cand['bank_index'],'matched_refs'] = bank_idx.loc[bank_idx['bank_index']==cand['bank_index'],'matched_refs'].apply(lambda lst: lst+[lrow.get('invoice_id')])
                    break
        if ledger.at[li,'status']=='Matched':
            continue
        # fallback amount-only (date tolerated)
        amt_candidates = candidates[abs(abs(candidates['amount'])-inv_amt) <= amount_tol]
        if not amt_candidates.empty:
            cand = amt_candidates.iloc[0]
            if lrow['type']=='AR' and cand['amount']>0:
                ledger.at[li,'matched_amount'] = cand['amount']; ledger.at[li,'status']='Matched'
                bank_idx.loc[bank_idx['bank_index']==cand['bank_index'],'matched']=True
                bank_idx.loc[bank_idx['bank_index']==cand['bank_index'],'matched_refs'] = bank_idx.loc[bank_idx['bank_index']==cand['bank_index'],'matched_refs'].apply(lambda lst: lst+[lrow.get('invoice_id')])
            if lrow['type']=='AP' and cand['amount']<0:
                ledger.at[li,'matched_amount'] = abs(cand['amount']); ledger.at[li,'status']='Matched'
                bank_idx.loc[bank_idx['bank_index']==cand['bank_index'],'matched']=True
                bank_idx.loc[bank_idx['bank_index']==cand['bank_index'],'matched_refs'] = bank_idx.loc[bank_idx['bank_index']==cand['bank_index'],'matched_refs'].apply(lambda lst: lst+[lrow.get('invoice_id')])
    # prepare outputs
    matched = ledger[ledger['status']=='Matched'].copy()
    unmatched_ledger = ledger[ledger['status']!='Matched'].copy()
    unmatched_bank = bank_idx[~bank_idx['matched']].copy()

    return matched, unmatched_ledger, unmatched_bank

def to_excel(dct):
    out = BytesIO()
    with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
        for k,v in dct.items():
            v.to_excel(writer, index=False, sheet_name=k[:31])
    return out.getvalue()

def reconciliation_module():
    st.title("🔄 Combined Reconciliation (AR/AP + Bank)")

    st.info("Upload one internal master ledger (may contain AR & AP) and a bank statement. The module will auto-split and reconcile; use filters to view AR or AP.")

    ledger_file = st.file_uploader("Internal Master Ledger (CSV/XLSX)", type=['csv','xlsx'])
    bank_file = st.file_uploader("Bank Statement (CSV/XLSX)", type=['csv','xlsx'])

    desc_threshold = st.sidebar.slider("Description match threshold", 50, 100, 80)
    date_tol = st.sidebar.number_input("Date tolerance (days)", 0, 30, 7)
    fuzzy_th = st.sidebar.slider("Fuzzy name threshold", 50, 100, 75)

    if not ledger_file or not bank_file:
        st.info("Upload both ledger and bank files to run reconciliation.")
        return

    ledger = load_file_flexible(ledger_file)
    bank = pd.read_csv(bank_file) if bank_file.name.endswith('.csv') else pd.read_excel(bank_file)

    matched, unmatched_ledger, unmatched_bank = reconcile_combined(ledger, bank, desc_threshold, date_tol, fuzzy_th)

    # Filters
    st.subheader("Filter Ledger Type")
    ledger_type = st.selectbox("Show", ['All','AR','AP'])
    if ledger_type != 'All':
        matched_show = matched[matched['type']==ledger_type]
        unmatched_show = unmatched_ledger[unmatched_ledger['type']==ledger_type]
    else:
        matched_show = matched
        unmatched_show = unmatched_ledger

    st.subheader("✅ Matched Ledger Items")
    st.dataframe(matched_show)

    st.subheader("⚠️ Unmatched Ledger Items")
    st.dataframe(unmatched_show)

    st.subheader("⚠️ Unmatched Bank Transactions")
    st.dataframe(unmatched_bank[['bank_index','date','description','amount']])

    # Download
    d = {'Matched': matched, 'Unmatched_Ledger': unmatched_ledger, 'Unmatched_Bank': unmatched_bank}
    st.download_button("⬇️ Download Reconciliation Excel", data=to_excel(d), file_name="combined_reconciliation.xlsx")

if __name__ == "__main__":
    reconciliation_module()
