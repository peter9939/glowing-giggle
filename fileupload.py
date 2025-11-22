import pandas as pd
import os
from io import BytesIO
from category_map import CATEGORY_PATTERNS, CATEGORY_MAP
from column_map import COLUMN_MAP
from rapidfuzz import fuzz, process
from smart_patterns import smart_dynamic_categorization
import streamlit as st

CATEGORY_FILE = "categorization.csv"
FUZZY_THRESHOLD = 85  # minimum score for fuzzy matching

# Create a normalized map from imported COLUMN_MAP (safe lookup)
NORMALIZED_COLUMN_MAP = {k.strip().lower(): v for k, v in COLUMN_MAP.items()}

# helpful alias sets derived from NORMALIZED_COLUMN_MAP
AMOUNT_ALIASES = {k for k, v in NORMALIZED_COLUMN_MAP.items() if v == 'Amount'}
DEBIT_ALIASES = {k for k, v in NORMALIZED_COLUMN_MAP.items() if v == 'Debit'}
CREDIT_ALIASES = {k for k, v in NORMALIZED_COLUMN_MAP.items() if v == 'Credit'}
DESCRIPTION_ALIASES = {k for k, v in NORMALIZED_COLUMN_MAP.items() if v == 'Description'}

# Add some common fallback aliases
AMOUNT_ALIASES.update({'amount','amt','value','transaction_value','payment_amount',
                       'paid','received','balance','closing_balance','opening_balance',
                       'total','net_amount','gross_amount','subtotal','grand_total'})
DEBIT_ALIASES.update({'debit','dr','withdrawal','debits'})
CREDIT_ALIASES.update({'credit','cr','deposit','credits'})
DESCRIPTION_ALIASES.update({'desc','description','details','memo','narration','particulars'})

# ---------------- Load or Create Categorization CSV ---------------- #
def load_categories():
    if os.path.exists(CATEGORY_FILE):
        try:
            df = pd.read_csv(CATEGORY_FILE)
            if df.empty:
                raise pd.errors.EmptyDataError
            return df
        except pd.errors.EmptyDataError:
            data = [[k, v[0], v[1]] for k, v in CATEGORY_MAP.items()]
            df = pd.DataFrame(data, columns=["Transaction Description", "Main Category", "Subcategory"])
            df.to_csv(CATEGORY_FILE, index=False)
            return df
    else:
        data = [[k, v[0], v[1]] for k, v in CATEGORY_MAP.items()]
        df = pd.DataFrame(data, columns=["Transaction Description", "Main Category", "Subcategory"])
        df.to_csv(CATEGORY_FILE, index=False)
        return df

categories_df = load_categories()

# ---------------- Fuzzy Matching ---------------- #
def fuzzy_match_category(description, choices_dict, threshold=FUZZY_THRESHOLD):
    if not choices_dict:
        return None, None, False, 0
    choices = [str(k).lower() for k in choices_dict.keys()]
    result = process.extractOne(description.lower(), choices, scorer=fuzz.token_sort_ratio)
    if result:
        match, score, _ = result
        if score >= threshold:
            main_cat, sub_cat = choices_dict[match]
            return main_cat, sub_cat, True, score
    return None, None, False, 0

# ---------------- Categorization ---------------- #
def categorize_flexible(description):
    desc = str(description).strip().lower()

    # Smart dynamic categorization
    main_cat, sub_cat, matched, confidence = smart_dynamic_categorization(desc)
    if matched:
        return main_cat, sub_cat, True, confidence

    # Exact match CATEGORY_MAP
    for key, value in CATEGORY_MAP.items():
        if desc == key.lower():
            return value[0], value[1], True, 100

    # Fuzzy match CATEGORY_MAP
    main_cat, sub_cat, matched, score = fuzzy_match_category(desc, CATEGORY_MAP)
    if matched:
        return main_cat, sub_cat, True, score

    # Exact match CATEGORY_PATTERNS
    if isinstance(CATEGORY_PATTERNS, dict):
        for key, value in CATEGORY_PATTERNS.items():
            if desc == key.lower() or key.lower() in desc:
                return value[0], value[1], True, 100
    elif isinstance(CATEGORY_PATTERNS, list):
        temp_dict = {str(p[0]).lower(): (p[1], p[2]) for p in CATEGORY_PATTERNS if len(p) >= 3}
        main_cat, sub_cat, matched, score = fuzzy_match_category(desc, temp_dict)
        if matched:
            return main_cat, sub_cat, True, score

    # User-edited categories
    match = categories_df['Transaction Description'].str.lower() == desc
    if match.any():
        row = categories_df[match].iloc[0]
        return row['Main Category'], row['Subcategory'], True, 100

    user_dict = {str(row['Transaction Description']).lower(): (row['Main Category'], row['Subcategory'])
                 for _, row in categories_df.iterrows()}
    main_cat, sub_cat, matched, score = fuzzy_match_category(desc, user_dict)
    if matched:
        return main_cat, sub_cat, True, score

    # Default fallback
    return "Uncategorized", "Uncategorized", False, 0

def safe_categorize(description):
    try:
        result = categorize_flexible(description)
        if not isinstance(result, (tuple, list)) or len(result) != 4:
            return "Uncategorized", "Uncategorized", False, 0
        return result
    except Exception as e:
        st.warning(f"Categorization failed for '{description}': {e}")
        return "Uncategorized", "Uncategorized", False, 0

# ---------------- Debit/Credit ---------------- #
def assign_dc(main_category):
    main_category = str(main_category).lower()
    if main_category in ["expense", "asset"]:
        return "Debit"
    elif main_category in ["revenue", "liability", "equity"]:
        return "Credit"
    else:
        return "Unknown"

# ---------------- AR/AP Detection ---------------- #
def detect_ar_ap(row):
    main_cat = str(row['Main Category']).lower()
    dc = str(row['Debit/Credit'])
    if main_cat == 'revenue' and dc == 'Credit':
        return 'Accounts Receivable'
    elif main_cat in ['expense', 'liability'] and dc == 'Debit':
        return 'Accounts Payable'
    else:
        return 'None'

# ---------------- File Upload Module ---------------- #
def file_upload_module():
    st.title("📊 Business Transaction Categorizer")

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])

    if not uploaded_file and 'processed_df' in st.session_state:
        df = st.session_state['processed_df'].copy()
        st.info("Loaded last processed dataset from session.")
    elif uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            if df.empty:
                df = pd.DataFrame([{'Description':'Sample Transaction','Amount':0}])
            st.success("File loaded successfully!")
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            return
    else:
        st.info("Upload a CSV/XLSX file or load previous session.")
        return

    # ---------------- Normalize & Rename columns ---------------- #
    df.columns = df.columns.str.strip().str.lower()
    df.rename(columns={col: NORMALIZED_COLUMN_MAP.get(col, col) for col in df.columns}, inplace=True)
    raw_df = df.copy()
    raw_df.columns = [str(c).strip() for c in raw_df.columns]

    # ---------------- Handle Amount ---------------- #
    cols_lower = {c: c.lower() for c in raw_df.columns}
    amount_like = [orig for orig, low in cols_lower.items() if low in AMOUNT_ALIASES or low == 'amount']

    if amount_like:
        src = amount_like[0]
        raw_df['Amount'] = pd.to_numeric(raw_df[src], errors='coerce').fillna(0)
    else:
        debit_cols = [orig for orig, low in cols_lower.items() if low in DEBIT_ALIASES]
        credit_cols = [orig for orig, low in cols_lower.items() if low in CREDIT_ALIASES]

        if debit_cols and credit_cols:
            raw_df['Amount'] = (
                pd.to_numeric(raw_df[debit_cols[0]], errors='coerce').fillna(0) -
                pd.to_numeric(raw_df[credit_cols[0]], errors='coerce').fillna(0)
            )
        elif debit_cols:
            raw_df['Amount'] = pd.to_numeric(raw_df[debit_cols[0]], errors='coerce').fillna(0)
        elif credit_cols:
            raw_df['Amount'] = -pd.to_numeric(raw_df[credit_cols[0]], errors='coerce').fillna(0)
        else:
            raw_df['Amount'] = 0.0

        for col in set(debit_cols + credit_cols):
            if col in raw_df.columns:
                raw_df.drop(columns=[col], inplace=True)

    raw_df['Balance Change'] = raw_df['Amount']

    # ---------------- Detect/Normalize Description Column ---------------- #
    desc_col = None
    for orig, low in cols_lower.items():
        if low in DESCRIPTION_ALIASES or 'desc' in low or 'description' in low:
            desc_col = orig
            break
    if desc_col:
        if desc_col != 'Description':
            raw_df.rename(columns={desc_col: 'Description'}, inplace=True)
    else:
        raw_df['Description'] = "No Description"

    # Drop duplicates safely
    raw_df = raw_df.drop_duplicates(subset=['Description', 'Amount'])

    # Remove old category columns if present
    for col in ['Main Category', 'Subcategory', 'Auto-Matched', 'Debit/Credit', 'Confidence', 'AR_AP', 'Category']:
        if col in raw_df.columns:
            raw_df.drop(columns=[col], inplace=True)

    # ---------------- Categorize Transactions (FIXED: no duplicate index) ---------------- #
    raw_df = raw_df.reset_index(drop=True)  # important
    results = raw_df['Description'].astype(str).apply(safe_categorize)
    raw_df['Main Category'] = results.apply(lambda r: r[0]).reset_index(drop=True)
    raw_df['Subcategory']   = results.apply(lambda r: r[1]).reset_index(drop=True)
    raw_df['Auto-Matched']  = results.apply(lambda r: r[2]).reset_index(drop=True)
    raw_df['Confidence']    = results.apply(lambda r: r[3]).reset_index(drop=True)
    raw_df['Debit/Credit']  = raw_df['Main Category'].apply(assign_dc)
    raw_df['AR_AP']         = raw_df.apply(detect_ar_ap, axis=1)

    st.session_state['processed_df'] = raw_df.copy()

    # ---------------- Filter / Search / Metrics / Downloads ---------------- #
    # Keep all your existing filter/search, metrics, and download code unchanged

    st.subheader("🔍 Filter Transactions")
    show_uncategorized = st.checkbox("Show only Uncategorized", value=False)
    main_categories = ["All"] + sorted(raw_df['Main Category'].dropna().unique().tolist())
    selected_main = st.selectbox("Filter by Main Category", main_categories)
    sub_categories = ["All"] + sorted(raw_df['Subcategory'].dropna().unique().tolist())
    selected_sub = st.selectbox("Filter by Subcategory", sub_categories)
    ar_ap_options = ["All", "Accounts Receivable", "Accounts Payable", "None"]
    selected_ar_ap = st.selectbox("Filter by AR/AP", ar_ap_options)

    filtered_df = raw_df.copy()
    if show_uncategorized:
        filtered_df = filtered_df[filtered_df['Auto-Matched'] == False]
    if selected_main != "All":
        filtered_df = filtered_df[filtered_df['Main Category'] == selected_main]
    if selected_sub != "All":
        filtered_df = filtered_df[filtered_df['Subcategory'] == selected_sub]
    if selected_ar_ap != "All":
        filtered_df = filtered_df[filtered_df['AR_AP'] == selected_ar_ap]

    st.subheader("🧾 Categorized Transactions")
    edited_df = st.data_editor(filtered_df, num_rows="dynamic")

    total_ar = raw_df[raw_df['AR_AP'] == 'Accounts Receivable']['Amount'].sum()
    total_ap = raw_df[raw_df['AR_AP'] == 'Accounts Payable']['Amount'].sum()
    st.metric("💰 Total Accounts Receivable", f"${total_ar:,.2f}")
    st.metric("📉 Total Accounts Payable", f"${total_ap:,.2f}")

    # Save edited categories
    if st.button("💾 Save Edited Categories"):
        changes = edited_df[['Description','Main Category','Subcategory']].copy()
        changes.rename(columns={'Description':'Transaction Description'}, inplace=True)
        existing = pd.read_csv(CATEGORY_FILE) if os.path.exists(CATEGORY_FILE) else pd.DataFrame(columns=['Transaction Description','Main Category','Subcategory'])
        for idx, row in changes.iterrows():
            existing = existing[existing['Transaction Description'] != row['Transaction Description']]
        updated = pd.concat([existing, changes], ignore_index=True)
        updated.to_csv(CATEGORY_FILE, index=False)
        global categories_df
        categories_df = load_categories()
        st.success("✅ Changes saved!")
        st.session_state['processed_df'] = raw_df.copy()

    # Downloads
    st.download_button("⬇️ Download CSV", data=edited_df.to_csv(index=False).encode('utf-8'), file_name="processed.csv")
    excel_output = BytesIO()
    with pd.ExcelWriter(excel_output, engine="xlsxwriter") as writer:
        edited_df.to_excel(writer,index=False)
    excel_output.seek(0)
    st.download_button("⬇️ Download Excel", data=excel_output, file_name="processed.xlsx")

    # Clear session
    if st.button("🧹 Clear Session"):
        if 'processed_df' in st.session_state:
            del st.session_state['processed_df']
        st.success("Session cleared.")
        st.rerun()

# ---------------- Helper for other modules ---------------- #
def get_processed_df():
    return st.session_state.get('processed_df', pd.DataFrame())

if __name__=="__main__":
    file_upload_module()
