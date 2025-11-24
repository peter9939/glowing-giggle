import streamlit as st
import pandas as pd
from io import BytesIO
import plotly.express as px

# ----------------------------------------- #
# LOAD DATA
# ----------------------------------------- #
def load_data():
    if 'processed_df' not in st.session_state:
        st.warning("No processed data found. Please upload your transactions first.")
        return pd.DataFrame()

    df = st.session_state['processed_df'].copy()

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').fillna(pd.Timestamp.today())
    else:
        df['Date'] = pd.Timestamp.today()

    needed = ['Main Category','Subcategory','Balance Change','Description']
    for col in needed:
        if col not in df.columns:
            df[col] = "Unknown"

    df['Balance Change'] = pd.to_numeric(df['Balance Change'], errors='coerce').fillna(0.0)
    return df


# ----------------------------------------- #
# VAT CALCULATION (REAL WORLD – GROSS AMOUNTS)
# ----------------------------------------- #
def extract_vat_from_gross(gross, rate):
    """
    Example: gross = 110, rate 10% => VAT = 10, Net = 100
    """
    vat = gross * (rate / (1 + rate))
    net = gross - vat
    return net, vat


# ----------------------------------------- #
# BUILD REAL JOURNAL ENTRIES FOR EACH ROW
# ----------------------------------------- #
def build_real_journal_entry(row):
    """
    Real accounting double-entry logic:
    
    REVENUE (sale):
        Dr Cash/AR (gross)
            Cr Revenue (net)
            Cr VAT Payable (vat)

    EXPENSE:
        Dr Expense (net)
        Dr VAT Receivable (vat)
            Cr Cash/AP (gross)
    """

    entries = []

    gross = row['Balance Change']
    net = row['Net Amount']
    vat = row['VAT Amount']
    cat = row['Main Category']

    if cat == "Revenue":
        # CASH RECEIVED FOR SALE
        entries.append({
            "Debit": "Accounts Receivable / Cash",
            "Credit": "Revenue",
            "Amount": net
        })
        if vat > 0:
            entries.append({
                "Debit": "Accounts Receivable / Cash",
                "Credit": "VAT Payable",
                "Amount": vat
            })

    elif cat == "Expense":
        # EXPENSE PURCHASE
        entries.append({
            "Debit": "Expense",
            "Credit": "Accounts Payable / Cash",
            "Amount": net
        })
        if vat > 0:
            entries.append({
                "Debit": "VAT Receivable",
                "Credit": "Accounts Payable / Cash",
                "Amount": vat
            })

    else:
        # Others default to cash movement
        entries.append({
            "Debit": "Cash/Bank" if gross > 0 else "Other",
            "Credit": "Other" if gross > 0 else "Cash/Bank",
            "Amount": abs(gross)
        })

    return entries


# ----------------------------------------- #
# APPLY VAT AND GENERATE JOURNALS
# ----------------------------------------- #
def apply_vat_and_journals(df, vat_rates):
    net_list = []
    vat_list = []
    journal_list = []

    for _, row in df.iterrows():
        rate = vat_rates.get(row['Main Category'], 0)
        gross = row['Balance Change']

        # Extract net + VAT from GROSS
        net, vat = extract_vat_from_gross(gross, rate)

        net_list.append(net)
        vat_list.append(vat)

        # Build journal entries
        row2 = row.copy()
        row2['Net Amount'] = net
        row2['VAT Amount'] = vat
        journal_list.append(build_real_journal_entry(row2))

    df['Net Amount'] = net_list
    df['VAT Amount'] = vat_list
    df['Journal Entries'] = journal_list

    return df


# ----------------------------------------- #
# PROFIT & INCOME TAX (REAL ACCOUNTING LOGIC)
# ----------------------------------------- #
def compute_profit_and_income_tax(df, tax_rate):
    total_rev = df[df['Main Category'] == "Revenue"]['Net Amount'].sum()
    total_exp = df[df['Main Category'] == "Expense"]['Net Amount'].sum()

    profit = total_rev - total_exp
    tax = profit * tax_rate if profit > 0 else 0

    return total_rev, total_exp, profit, tax


# ----------------------------------------- #
# EXPORT TO EXCEL
# ----------------------------------------- #
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
# ----------------------------------------- #
# STREAMLIT APP UI
# ----------------------------------------- #
def tax_module():
    st.title("💰 Advanced Real Accounting System (VAT + Income Tax + Journals)")

    # Load data from session
    df = load_data()
    if df.empty:
        return

    # ---------------- Sidebar Filters ---------------- #
    st.sidebar.header("Filters")
    start_date = st.sidebar.date_input("Start Date", df['Date'].min())
    end_date = st.sidebar.date_input("End Date", df['Date'].max())

    main_categories = st.sidebar.multiselect("Main Category", df['Main Category'].unique())
    sub_categories = st.sidebar.multiselect("Subcategory", df['Subcategory'].unique())

    df_filtered = df[
        (df['Date'] >= pd.to_datetime(start_date)) &
        (df['Date'] <= pd.to_datetime(end_date))
    ]

    if main_categories:
        df_filtered = df_filtered[df_filtered['Main Category'].isin(main_categories)]
    if sub_categories:
        df_filtered = df_filtered[df_filtered['Subcategory'].isin(sub_categories)]

    if df_filtered.empty:
        st.info("No transactions found for selected filters.")
        return

    # ---------------- VAT RATES (editable) ---------------- #
    st.subheader("🛠 VAT Rates by Category (Real Accounting)")
    vat_rates = {}
    for cat in df_filtered['Main Category'].unique():
        vat_rates[cat] = st.number_input(
            f"VAT Rate for {cat} (%)",
            min_value=0.0,
            max_value=100.0,
            value=10.0 # default 10%
        ) / 100

    # ---------------- APPLY VAT & JOURNALS ---------------- #
    df_taxed = apply_vat_and_journals(df_filtered, vat_rates)

    # ---------------- KPIs: VAT Summary ---------------- #
    st.subheader("📊 VAT Summary")
    total_vat = df_taxed['VAT Amount'].sum()
    total_net = df_taxed['Net Amount'].sum()
    total_gross = df_taxed['Balance Change'].sum()

    colA, colB, colC = st.columns(3)
    colA.metric("Total Gross", f"${total_gross:,.2f}")
    colB.metric("Total Net", f"${total_net:,.2f}")
    colC.metric("Total VAT", f"${total_vat:,.2f}")

    # ---------------- Category Summary ---------------- #
    st.subheader("📈 Summary by Main Category")
    summary = df_taxed.groupby("Main Category")[["Net Amount", "VAT Amount"]].sum().reset_index()
    st.dataframe(summary)

    fig = px.bar(summary, x="Main Category", y=["Net Amount", "VAT Amount"], text_auto=True,
                 title="Net Amount vs VAT per Category")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- Profit & Income Tax ---------------- #
    st.subheader("💼 Profit & Income Tax (Real Accounting)")

    income_tax_rate = st.number_input(
        "Income Tax Rate (%)",
        min_value=0.0, max_value=100.0, value=20.0
    ) / 100

    total_rev, total_exp, profit, income_tax = compute_profit_and_income_tax(df_taxed, income_tax_rate)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Revenue (Net)", f"${total_rev:,.2f}")
    col2.metric("Expense (Net)", f"${total_exp:,.2f}")
    col3.metric("Profit", f"${profit:,.2f}")
    col4.metric("Income Tax", f"${income_tax:,.2f}")

    # ---------------- Income Tax Journal Entry ---------------- #
    st.subheader("📘 Income Tax Journal Entry")

    if profit > 0:
        st.write("""
        **Dr Income Tax Expense**  
        **Cr Income Tax Payable**
        """)
        st.write(f"**Amount:** ${income_tax:,.2f}")
    else:
        st.write("No income tax (profit is zero or negative).")

    # ---------------- Show Journal Entries ---------------- #
    st.subheader("📚 Journal Entries (Per Transaction)")

    for i, row in df_taxed.iterrows():
        st.markdown(f"### Transaction {i+1}: {row['Description']}")
        st.write(f"**Category:** {row['Main Category']} — **Gross:** {row['Balance Change']:,.2f}")
        st.write("#### Journal Entries:")
        journals = row['Journal Entries']

        journal_df = pd.DataFrame(journals)
        st.dataframe(journal_df)

    # ---------------- Download Excel ---------------- #
    st.download_button(
        "⬇️ Download Tax Report (Excel)",
        data=to_excel(df_taxed),
        file_name="real_accounting_tax_report.xlsx"
    )


# ----------------------------------------- #
# MAIN
# ----------------------------------------- #
if __name__ == "__main__":
    tax_module()
