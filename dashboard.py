# dashboard_module.py
import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
from fpdf import FPDF
from fileupload import get_processed_df  # uses your existing fileupload.py

# ---------------- Helpers: load & normalize ---------------- #
def load_data():
    """
    Load processed_df from fileupload module and normalize columns.
    Ensures Date, Main Category, Subcategory, Balance Change, Description, Auto-Matched exist.
    """
    if 'processed_df' not in st.session_state:
        st.error("No processed data found. Please upload your transactions first in Upload Transactions.")
        return pd.DataFrame()

    df = st.session_state['processed_df'].copy()

    # Ensure Date column
    if 'Date' not in df.columns:
        df['Date'] = pd.Timestamp.today()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').fillna(pd.Timestamp.today())

    # Ensure essential columns exist
    essential_cols = ['Main Category', 'Subcategory', 'Balance Change', 'Description', 'Auto-Matched', 'Date']
    for col in essential_cols:
        if col not in df.columns:
            if col == 'Balance Change':
                df[col] = 0.0
            elif col == 'Auto-Matched':
                df[col] = False
            elif col == 'Date':
                df[col] = pd.Timestamp.today()
            else:
                df[col] = 'Unknown'

    # Normalize column types
    df['Balance Change'] = pd.to_numeric(df['Balance Change'], errors='coerce').fillna(0.0)
    df['Description'] = df['Description'].astype(str)
    df['Main Category'] = df['Main Category'].astype(str)
    df['Subcategory'] = df['Subcategory'].astype(str)

    # Keep only the essentials (but preserve others if present)
    # We'll return full df but guarantee these columns exist
    return df

# ---------------- KPIs ---------------- #
def compute_kpis(df):
    revenue = df.loc[df['Main Category'].str.lower() == 'revenue', 'Balance Change'].sum()
    expense = df.loc[df['Main Category'].str.lower() == 'expense', 'Balance Change'].sum()
    net_profit = revenue - expense
    total_assets = df.loc[df['Main Category'].str.lower() == 'asset', 'Balance Change'].sum()
    total_liabilities = df.loc[df['Main Category'].str.lower() == 'liability', 'Balance Change'].sum()
    total_equity = df.loc[df['Main Category'].str.lower() == 'equity', 'Balance Change'].sum()
    current_ratio = (total_assets / total_liabilities) if total_liabilities != 0 else float('inf')
    return {
        'Revenue': revenue,
        'Expense': expense,
        'Net Profit': net_profit,
        'Total Assets': total_assets,
        'Total Liabilities': total_liabilities,
        'Total Equity': total_equity,
        'Current Ratio': current_ratio
    }

# ---------------- Export Functions ---------------- #
def to_excel(df_dict):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in df_dict.items():
            df.to_excel(writer, index=False, sheet_name=sheet_name)
            worksheet = writer.sheets[sheet_name]
            # Set reasonable column widths
            for i, col in enumerate(df.columns):
                try:
                    col_width = max(df[col].astype(str).map(len).max(), len(col)) + 2
                except Exception:
                    col_width = len(col) + 2
                worksheet.set_column(i, i, col_width)
    output.seek(0)
    return output.getvalue()

def export_pdf(df_dict, kpis):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, 'Professional Accounting Dashboard Report', ln=True, align='C')
    pdf.ln(6)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, 'Key Performance Indicators', ln=True)
    pdf.set_font("Arial", '', 10)
    for k, v in kpis.items():
        if isinstance(v, (int, float)):
            text = f'{k}: ${v:,.2f}'
        else:
            text = f'{k}: {v}'
        pdf.cell(0, 8, text, ln=True)
    pdf.ln(6)

    for name, df in df_dict.items():
        if df.empty:
            continue
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, name, ln=True)
        pdf.set_font("Arial", '', 8)
        # table header
        col_count = len(df.columns)
        page_width = pdf.w - 2 * pdf.l_margin
        col_w = page_width / max(col_count, 1)
        th = 6
        # header row
        for col in df.columns:
            pdf.cell(col_w, th, str(col)[:20], border=1)
        pdf.ln(th)
        # rows (capped length per cell)
        for row in df.values.tolist():
            for item in row:
                text = str(item).replace("–", "-").replace("—", "-")
                pdf.cell(col_w, th, text[:20], border=1)
            pdf.ln(th)
        pdf.ln(4)

    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output.read()

# ---------------- Journal & Ledger generation (CORRECTED double-entry) ---------------- #
def generate_journal_entries(df):
    """
    Generate proper double-entry journal entries using Option A:
    - Debit/Credit pairs are auto generated using Cash / Bank as the counter account.
    - For unknown categories, use 'Suspense' as counteraccount.
    """
    journal_entries = []
    # Normalize main category for matching
    df = df.copy()
    df['main_lower'] = df['Main Category'].str.strip().str.lower()

    for _, row in df.iterrows():
        amount = float(row['Balance Change'] or 0.0)
        # Use absolute magnitude for ledger entries; direction determined by category logic
        amt = abs(amount)
        date = row['Date']
        desc = row.get('Description', '') or ''
        main_cat = row.get('main_lower', '')

        # Set default accounts
        cash_account = "Cash / Bank"
        suspense_account = "Suspense"

        # Revenue: Credit revenue, Debit Cash/Bank
        if main_cat == 'revenue':
            journal_entries.append({
                "Date": date,
                "Description": desc,
                "Debit Account": cash_account,
                "Debit": amt,
                "Credit Account": row['Main Category'] + " - " + row['Subcategory'],
                "Credit": amt
            })
        # Expense: Debit Expense, Credit Cash/Bank
        elif main_cat == 'expense':
            journal_entries.append({
                "Date": date,
                "Description": desc,
                "Debit Account": row['Main Category'] + " - " + row['Subcategory'],
                "Debit": amt,
                "Credit Account": cash_account,
                "Credit": amt
            })
        # Asset increase: Debit Asset, Credit Cash/Bank (e.g., equipment purchased -> asset increases and cash decreases)
        elif main_cat == 'asset':
            journal_entries.append({
                "Date": date,
                "Description": desc,
                "Debit Account": row['Main Category'] + " - " + row['Subcategory'],
                "Debit": amt,
                "Credit Account": cash_account,
                "Credit": amt
            })
        # Liability increase (e.g., loan received): Credit Liability, Debit Cash/Bank
        elif main_cat == 'liability':
            journal_entries.append({
                "Date": date,
                "Description": desc,
                "Debit Account": cash_account,
                "Debit": amt,
                "Credit Account": row['Main Category'] + " - " + row['Subcategory'],
                "Credit": amt
            })
        # Equity increase: Credit Equity, Debit Cash/Bank
        elif main_cat == 'equity':
            journal_entries.append({
                "Date": date,
                "Description": desc,
                "Debit Account": cash_account,
                "Debit": amt,
                "Credit Account": row['Main Category'] + " - " + row['Subcategory'],
                "Credit": amt
            })
        # Uncategorized or unknown: put amount to Suspense vs Cash/Bank (safe default)
        else:
            # If amount is positive — assume Cash increase, credit Suspense
            journal_entries.append({
                "Date": date,
                "Description": desc,
                "Debit Account": cash_account,
                "Debit": amt,
                "Credit Account": suspense_account,
                "Credit": amt
            })

    journal_df = pd.DataFrame(journal_entries)
    # Ensure numeric types
    if not journal_df.empty:
        journal_df['Debit'] = pd.to_numeric(journal_df['Debit'], errors='coerce').fillna(0.0)
        journal_df['Credit'] = pd.to_numeric(journal_df['Credit'], errors='coerce').fillna(0.0)
    return journal_df

def build_ledger_from_journal(journal_df):
    """
    Convert journal entries into ledger rows (debits and credits as separate rows)
    and compute running balances per account.
    """
    ledger_rows = []
    for _, row in journal_df.iterrows():
        # Debit row
        ledger_rows.append({
            "Account": row['Debit Account'],
            "Date": row['Date'],
            "Description": row['Description'],
            "Debit": row['Debit'],
            "Credit": 0.0
        })
        # Credit row
        ledger_rows.append({
            "Account": row['Credit Account'],
            "Date": row['Date'],
            "Description": row['Description'],
            "Debit": 0.0,
            "Credit": row['Credit']
        })

    ledger_df = pd.DataFrame(ledger_rows)
    if ledger_df.empty:
        return ledger_df

    # Normalize numeric columns
    ledger_df['Debit'] = pd.to_numeric(ledger_df['Debit'], errors='coerce').fillna(0.0)
    ledger_df['Credit'] = pd.to_numeric(ledger_df['Credit'], errors='coerce').fillna(0.0)

    ledger_df.sort_values(['Account', 'Date'], inplace=True)
    # Running balance per account: cumulative (Debit - Credit)
    ledger_df['Running Balance'] = ledger_df.groupby('Account').apply(
        lambda g: (g['Debit'] - g['Credit']).cumsum()
    ).reset_index(level=0, drop=True)
    return ledger_df

# ---------------- Dashboard Module ---------------- #
def dashboard_module():
    st.title("📊 Professional Accounting Dashboard (Production Ready)")

    df = load_data()
    if df.empty:
        return

    # Sidebar Filters
    st.sidebar.header("Filters")
    start_date = st.sidebar.date_input("Start Date", df['Date'].min().date())
    end_date = st.sidebar.date_input("End Date", df['Date'].max().date())
    main_categories = st.sidebar.multiselect("Main Category", sorted(df['Main Category'].unique().tolist()))
    sub_categories = st.sidebar.multiselect("Subcategory", sorted(df['Subcategory'].unique().tolist()))
    keyword = st.sidebar.text_input("Search Description")

    # Apply filters
    df_filtered = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
    if main_categories:
        df_filtered = df_filtered[df_filtered['Main Category'].isin(main_categories)]
    if sub_categories:
        df_filtered = df_filtered[df_filtered['Subcategory'].isin(sub_categories)]
    if keyword:
        df_filtered = df_filtered[df_filtered['Description'].str.contains(keyword, case=False, na=False)]

    if df_filtered.empty:
        st.warning("No transactions found for the selected filters.")
        return

    # KPIs
    kpis = compute_kpis(df_filtered)
    st.subheader("Key Performance Indicators")
    kcol1, kcol2, kcol3 = st.columns(3)
    kcol1.metric("Revenue", f"${kpis['Revenue']:,.2f}")
    kcol2.metric("Expense", f"${kpis['Expense']:,.2f}")
    kcol3.metric("Net Profit", f"${kpis['Net Profit']:,.2f}")
    kcol4, kcol5, kcol6 = st.columns(3)
    kcol4.metric("Total Assets", f"${kpis['Total Assets']:,.2f}")
    kcol5.metric("Total Liabilities", f"${kpis['Total Liabilities']:,.2f}")
    kcol6.metric("Total Equity", f"${kpis['Total Equity']:,.2f}")
    try:
        cr_display = f"{kpis['Current Ratio']:.2f}" if kpis['Current Ratio'] != float('inf') else "∞"
    except Exception:
        cr_display = "N/A"
    st.metric("Current Ratio", cr_display)

    # Tabs
    tabs = st.tabs(["Trial Balance", "Income Statement", "Balance Sheet", "Cash Flow", "P&L", "Category Drilldown", "Journal & Ledger"])

    # ---------- Trial Balance ----------
    with tabs[0]:
        st.subheader("Trial Balance")
        trial_cols = ['Main Category', 'Subcategory', 'Balance Change']
        trial = df_filtered[trial_cols].copy()
        # produce Debit/Credit columns according to main category
        def calc_debit_credit(r):
            mc = str(r['Main Category']).strip().lower()
            amt = float(r['Balance Change'])
            if mc == 'expense' or mc == 'asset':
                # treat as debit
                return pd.Series([abs(amt), 0.0])
            elif mc == 'revenue' or mc == 'liability' or mc == 'equity':
                return pd.Series([0.0, abs(amt)])
            else:
                # unknown: put in debit
                return pd.Series([max(amt, 0.0), max(-amt, 0.0)])
        trial[['Debit', 'Credit']] = trial.apply(calc_debit_credit, axis=1)
        trial['Status'] = trial.apply(lambda r: 'Balanced' if abs(r['Debit'] - r['Credit']) < 0.01 else 'Check', axis=1)
        st.dataframe(trial[['Main Category', 'Subcategory', 'Debit', 'Credit', 'Balance Change', 'Status']])

        # Main Category Chart
        trial_summary = trial.groupby('Main Category')[['Debit', 'Credit']].sum().reset_index()
        if not trial_summary.empty:
            fig = px.bar(trial_summary, x='Main Category', y=['Debit', 'Credit'], barmode='group',
                         title='Trial Balance by Main Category', text_auto='.2f')
            st.plotly_chart(fig, use_container_width=True)

        total_debit = trial['Debit'].sum()
        total_credit = trial['Credit'].sum()
        if abs(total_debit - total_credit) < 0.01:
            st.success(f"✅ Total Trial Balance is balanced (Debit={total_debit:,.2f}, Credit={total_credit:,.2f})")
        else:
            st.error(f"⚠️ Total Trial Balance NOT balanced! Debit={total_debit:,.2f}, Credit={total_credit:,.2f}")

    # ---------- Income Statement ----------
    with tabs[1]:
        st.subheader("Income Statement")
        income = df_filtered[df_filtered['Main Category'].str.lower().isin(['revenue', 'expense'])]
        if not income.empty:
            main_summary = income.groupby('Main Category')['Balance Change'].sum().reset_index()
            st.markdown("**By Main Category**")
            st.dataframe(main_summary)
            fig_main = px.bar(main_summary, x='Main Category', y='Balance Change', color='Main Category',
                              title='Income Statement - Main Category', text='Balance Change')
            fig_main.update_traces(texttemplate='%{text:,.2f}', textposition='outside')
            st.plotly_chart(fig_main, use_container_width=True)

            sub_summary = income.groupby('Subcategory')['Balance Change'].sum().reset_index()
            st.markdown("**By Subcategory**")
            st.dataframe(sub_summary)
            fig_sub = px.bar(sub_summary, x='Subcategory', y='Balance Change', color='Subcategory',
                             title='Income Statement - Subcategory', text='Balance Change')
            fig_sub.update_traces(texttemplate='%{text:,.2f}', textposition='outside')
            st.plotly_chart(fig_sub, use_container_width=True)

    # ---------- Balance Sheet ----------
    with tabs[2]:
        st.subheader("Balance Sheet")
        bs = df_filtered[df_filtered['Main Category'].str.lower().isin(['asset', 'liability', 'equity'])]
        categories = ['Asset', 'Liability', 'Equity']
        bs_summary = bs.groupby('Main Category')['Balance Change'].sum().reindex(categories, fill_value=0).reset_index()
        st.dataframe(bs_summary)
        fig_bs = px.bar(bs_summary, x='Main Category', y='Balance Change', color='Main Category',
                        title='Balance Sheet Overview')
        st.plotly_chart(fig_bs, use_container_width=True)
        # Balance check
        if abs(kpis['Total Assets'] - (kpis['Total Liabilities'] + kpis['Total Equity'])) < 0.01:
            st.success("✅ Balance Sheet is balanced (Assets = Liabilities + Equity)")
        else:
            st.error("⚠️ Balance Sheet NOT balanced!")

    # ---------- Cash Flow ----------
    with tabs[3]:
        st.subheader("Cash Flow")
        cash_df = df_filtered[df_filtered['Main Category'].str.lower().isin(['revenue', 'expense', 'liability', 'equity'])]
        if not cash_df.empty:
            cash_df = cash_df.copy()
            # classify type: revenue -> cash in, expense -> cash out; liability/equity movements treated as financing
            def classify_cash(x):
                mc = x['Main Category'].strip().lower()
                if mc == 'revenue':
                    return 'Cash In (Revenue)'
                if mc == 'expense':
                    return 'Cash Out (Expense)'
                if mc == 'liability':
                    return 'Cash In/Out (Liability)'
                if mc == 'equity':
                    return 'Cash In/Out (Equity)'
                return 'Other'
            cash_df['Type'] = cash_df.apply(classify_cash, axis=1)
            cash_df['Month'] = cash_df['Date'].dt.to_period('M').astype(str)
            cash_flow = cash_df.groupby(['Month', 'Type'])['Balance Change'].sum().reset_index()
            st.dataframe(cash_flow)
            fig_cash = px.bar(cash_flow, x='Month', y='Balance Change', color='Type', barmode='group',
                              title='Monthly Cash Flow (AR vs AP)', text='Balance Change')
            fig_cash.update_traces(texttemplate='%{text:,.2f}', textposition='outside')
            st.plotly_chart(fig_cash, use_container_width=True)

    # ---------- P&L ----------
    with tabs[4]:
        st.subheader("Profit & Loss Statement")
        pnl = df_filtered[df_filtered['Main Category'].str.lower().isin(['revenue', 'expense'])]
        if not pnl.empty:
            pnl = pnl.copy()
            pnl['Month'] = pnl['Date'].dt.to_period('M').astype(str)
            pnl_summary = pnl.groupby(['Month', 'Main Category'])['Balance Change'].sum().unstack().fillna(0)
            # Ensure Revenue and Expense exist
            pnl_summary['Revenue'] = pnl_summary.get('Revenue', 0)
            pnl_summary['Expense'] = pnl_summary.get('Expense', 0)
            pnl_summary['Net Profit'] = pnl_summary['Revenue'] - pnl_summary['Expense']
            pnl_summary = pnl_summary.reset_index()
            st.dataframe(pnl_summary)
            fig_pnl = px.bar(pnl_summary, x='Month', y=['Revenue', 'Expense'], barmode='group', title='Monthly Revenue vs Expense', text_auto='.2f')
            st.plotly_chart(fig_pnl, use_container_width=True)
            fig_net = px.line(pnl_summary, x='Month', y='Net Profit', markers=True, title='Monthly Net Profit Trend')
            st.plotly_chart(fig_net, use_container_width=True)

    # ---------- Category Drilldown ----------
    with tabs[5]:
        st.subheader("Category Drilldown")
        main_cat_summary = df_filtered.groupby('Main Category')['Balance Change'].sum().reset_index()
        if not main_cat_summary.empty:
            fig = px.bar(main_cat_summary, x='Main Category', y='Balance Change', color='Main Category',
                         title="Main Category Overview", text='Balance Change')
            fig.update_traces(texttemplate='%{text:,.2f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

        selected_main_cat = st.selectbox("Select Main Category for Drilldown", [''] + main_cat_summary['Main Category'].tolist())
        if selected_main_cat:
            subcat_df = df_filtered[df_filtered['Main Category'] == selected_main_cat]
            subcat_summary = subcat_df.groupby('Subcategory')['Balance Change'].sum().reset_index()
            fig2 = px.bar(subcat_summary, x='Subcategory', y='Balance Change', color='Subcategory',
                          title=f"Subcategories under {selected_main_cat}", text='Balance Change')
            fig2.update_traces(texttemplate='%{text:,.2f}', textposition='outside')
            st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(subcat_summary)

    # ---------- Journal & Ledger ----------
    with tabs[6]:
        st.subheader("📘 Journal & Ledger")
        # Prepare journal entries using corrected double-entry mappings
        jdf = df_filtered.copy()
        # Create a human-readable account name for each row if not present
        if 'Account' not in jdf.columns:
            jdf['Account'] = jdf['Main Category'].astype(str) + " - " + jdf['Subcategory'].astype(str)
        if 'Description' not in jdf.columns:
            jdf['Description'] = "No Description"

        journal_df = generate_journal_entries(jdf)
        st.subheader("📄 Journal Entries")
        if not journal_df.empty:
            # reorder columns for readability
            cols_order = ['Date', 'Description', 'Debit Account', 'Debit', 'Credit Account', 'Credit']
            cols_present = [c for c in cols_order if c in journal_df.columns]
            st.dataframe(journal_df[cols_present])
        else:
            st.info("No journal entries generated.")

        # Build ledger from journal
        ledger_df = build_ledger_from_journal(journal_df)
        st.subheader("📚 Ledger")
        if not ledger_df.empty:
            st.dataframe(ledger_df)
            summary = ledger_df.groupby("Account").agg({"Debit": "sum", "Credit": "sum"}).reset_index()
            summary["Closing Balance"] = summary["Debit"] - summary["Credit"]
            st.subheader("📘 Ledger Summary")
            st.dataframe(summary)
            # Running Balance chart selector
            st.subheader("📈 Ledger Balance Charts")
            selected_acct = st.selectbox("Select Account", summary["Account"].unique())
            if selected_acct:
                acct_df = ledger_df[ledger_df["Account"] == selected_acct]
                fig = px.line(acct_df, x="Date", y="Running Balance", title=f"Running Balance - {selected_acct}", markers=True)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Ledger is empty (no journal entries).")

    # ---------------- Download Reports ---------------- #
    st.markdown("---")
    st.subheader("📥 Download Reports")
    df_dict = {'Raw Data': df_filtered}
    # include main category summary if available
    if 'main_cat_summary' in locals() and not main_cat_summary.empty:
        df_dict['Main Category Summary'] = main_cat_summary
    if 'selected_main_cat' in locals() and selected_main_cat:
        df_dict[f'{selected_main_cat} Subcategories'] = subcat_summary
    st.download_button("⬇️ Download Excel Report", data=to_excel(df_dict), file_name="dashboard_report.xlsx")
    st.download_button("⬇️ Download PDF Report", data=export_pdf(df_dict, kpis), file_name="dashboard_report.pdf")


if __name__ == "__main__":
    dashboard_module()
