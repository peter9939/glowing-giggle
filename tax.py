import streamlit as st
import pandas as pd
from io import BytesIO
import plotly.express as px

# ---------------- Load Processed Data ---------------- #
def load_data():
    if 'processed_df' not in st.session_state:
        st.warning("No processed data found. Please upload your transactions first.")
        return pd.DataFrame()

    df = st.session_state['processed_df'].copy()

    # Ensure Date exists and is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').fillna(pd.Timestamp.today())
    else:
        df['Date'] = pd.Timestamp.today()

    # Ensure essential columns exist
    for col in ['Main Category','Subcategory','Balance Change','Account','Description']:
        if col not in df.columns:
            df[col] = 0 if col == 'Balance Change' else 'Unknown'

    df['Balance Change'] = pd.to_numeric(df['Balance Change'], errors='coerce').fillna(0.0)
    return df

# ---------------- Default Tax & VAT Rates ---------------- #
DEFAULT_TAX_RATES = {
    'Revenue': 0.10,    # 10% standard VAT on revenue
    'Expense': 0.10,    # Assume 10% reclaimable VAT on some expenses
    'Asset': 0.10,      # VAT may apply to assets
    'Liability': 0.0,
    'Equity': 0.0,
    'Uncategorized': 0.0
}

# ---------------- VAT Detection ---------------- #
def detect_vat(row, vat_rate=0.10):
    """
    Auto detect VAT on expenses/assets: 
    Revenue VAT = liability
    Expense VAT = recoverable (asset)
    """
    main_cat = row['Main Category']
    amt = row['Balance Change']

    if main_cat == 'Revenue':
        tax_amt = amt * vat_rate
        net_amt = amt - tax_amt
        vat_account = 'VAT Payable'
    elif main_cat in ['Expense','Asset']:
        tax_amt = amt * vat_rate
        net_amt = amt - tax_amt
        vat_account = 'VAT Receivable'
    else:
        tax_amt = 0
        net_amt = amt
        vat_account = 'None'

    return pd.Series([net_amt, tax_amt, vat_account])

# ---------------- Tax Calculation & Journals ---------------- #
def calculate_tax_and_journals(df, tax_rates):
    df['Tax Rate'] = df['Main Category'].map(lambda x: tax_rates.get(x, 0))
    
    # Calculate Net/Tax and VAT accounts
    df[['Net Amount','Tax Amount','VAT Account']] = df.apply(lambda r: detect_vat(r, r['Tax Rate']), axis=1)

    # Determine AR/AP logic
    df['Account Type'] = df.apply(lambda r: 'Accounts Receivable' if r['Main Category'] == 'Revenue' else
                                              ('Accounts Payable' if r['Main Category'] == 'Expense' else 'Cash/Bank'), axis=1)

    # Auto-generate simplified journal entries
    df['Debit Account'] = df.apply(lambda r: 
                                   r['Account Type'] if r['Main Category'] in ['Revenue','Expense'] else 'Cash/Bank', axis=1)
    
    df['Credit Account'] = df.apply(lambda r: 
                                    ('Revenue' if r['Main Category']=='Revenue' else
                                     ('Expense' if r['Main Category']=='Expense' else 'Cash/Bank')), axis=1)
    
    # Adjust for VAT: create additional journal entries for VAT accounts
    df['Debit Account VAT'] = df.apply(lambda r: r['VAT Account'] if r['VAT Account']=='VAT Receivable' else None, axis=1)
    df['Credit Account VAT'] = df.apply(lambda r: r['VAT Account'] if r['VAT Account']=='VAT Payable' else None, axis=1)

    return df

# ---------------- Export to Excel ---------------- #
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Tax Report')
        worksheet = writer.sheets['Tax Report']
        for i, col in enumerate(df.columns):
            col_width = max(df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(i, i, col_width)
    output.seek(0)
    return output.getvalue()

# ---------------- Tax Module ---------------- #
def tax_module():
    st.title("💰 Advanced Tax & VAT Management")

    df = load_data()
    if df.empty:
        return

    # ---------------- Sidebar Filters ---------------- #
    st.sidebar.header("Filters")
    start_date = st.sidebar.date_input("Start Date", df['Date'].min())
    end_date = st.sidebar.date_input("End Date", df['Date'].max())
    main_categories = st.sidebar.multiselect("Main Category", df['Main Category'].unique())
    sub_categories = st.sidebar.multiselect("Subcategory", df['Subcategory'].unique())
    accounts = st.sidebar.multiselect("Accounts", df['Account'].unique())

    df_filtered = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
    if main_categories:
        df_filtered = df_filtered[df_filtered['Main Category'].isin(main_categories)]
    if sub_categories:
        df_filtered = df_filtered[df_filtered['Subcategory'].isin(sub_categories)]
    if accounts:
        df_filtered = df_filtered[df_filtered['Account'].isin(accounts)]

    if df_filtered.empty:
        st.info("No transactions found for selected filters.")
        return

    # ---------------- Editable Tax Rates ---------------- #
    st.subheader("🛠 Editable Tax Rates")
    tax_rates = {}
    for cat in df_filtered['Main Category'].unique():
        tax_rates[cat] = st.number_input(
            f"Tax Rate for {cat} (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(DEFAULT_TAX_RATES.get(cat, 0) * 100)
        ) / 100

    # ---------------- Calculate Tax & Journals ---------------- #
    df_taxed = calculate_tax_and_journals(df_filtered, tax_rates)

    # ---------------- KPIs ---------------- #
    st.subheader("📊 Tax & VAT KPIs")
    total_txn = len(df_taxed)
    total_tax = df_taxed['Tax Amount'].sum()
    total_net = df_taxed['Net Amount'].sum()
    cols = st.columns(3)
    cols[0].metric("Total Transactions", total_txn)
    cols[1].metric("Total Tax Amount", f"${total_tax:,.2f}")
    cols[2].metric("Total Net Amount", f"${total_net:,.2f}")

    # ---------------- Tax Summary ---------------- #
    st.subheader("📈 Tax Summary by Main Category")
    tax_summary = df_taxed.groupby('Main Category')[['Tax Amount','Net Amount']].sum().reset_index()
    st.dataframe(tax_summary)

    # ---------------- Charts ---------------- #
    fig_bar = px.bar(tax_summary, x='Main Category', y=['Net Amount','Tax Amount'],
                     text_auto=True, title='Net vs Tax per Category',
                     color_discrete_sequence=px.colors.qualitative.Bold)
    st.plotly_chart(fig_bar, use_container_width=True)

    fig_pie = px.pie(tax_summary, names='Main Category', values='Tax Amount', title='Tax Share by Category')
    st.plotly_chart(fig_pie, use_container_width=True)

    # ---------------- Subcategory Drilldown ---------------- #
    st.subheader("🔍 Subcategory Drilldown")
    selected_main_cat = st.selectbox("Select Main Category for Drilldown", [''] + df_taxed['Main Category'].unique().tolist())
    if selected_main_cat:
        subcat_df = df_taxed[df_taxed['Main Category'] == selected_main_cat]
        subcat_summary = subcat_df.groupby('Subcategory')[['Tax Amount','Net Amount']].sum().reset_index()
        st.dataframe(subcat_summary)

        fig_sub = px.bar(subcat_summary, x='Subcategory', y=['Net Amount','Tax Amount'], text_auto=True,
                         title=f"Net vs Tax by Subcategory for {selected_main_cat}",
                         color_discrete_sequence=px.colors.qualitative.Vivid)
        st.plotly_chart(fig_sub, use_container_width=True)

    # ---------------- Download ---------------- #
    st.download_button(
        "⬇️ Download Tax Report (Excel)",
        data=to_excel(df_taxed),
        file_name="advanced_tax_report.xlsx"
    )

# ---------------- Main ---------------- #
if __name__=="__main__":
    tax_module()
