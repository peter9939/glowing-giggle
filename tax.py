import streamlit as st
import pandas as pd
from fpdf import FPDF
import plotly.express as px
import plotly.io as pio
import io
import re

# ---------------- LOAD DATA ---------------- #
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
    return df

# ---------------- PREPROCESS AMOUNTS & CURRENCY ---------------- #
def preprocess_amounts(df):
    df['Currency'] = df['Balance Change'].astype(str).str.extract(r'([A-Z$€£₹¥]+)')[0].fillna('$')
    df['Balance Change'] = df['Balance Change'].astype(str).str.replace(r"[^\d\.-]", "", regex=True)
    df['Balance Change'] = pd.to_numeric(df['Balance Change'], errors='coerce').fillna(0.0)
    return df

# ---------------- VAT / NET CALC ---------------- #
def extract_vat_from_gross(gross, rate, category):
    """
    VAT applies only to Revenue (sales) and Expense (purchases)
    """
    if category not in ["Revenue", "Expense"]:
        return gross, 0.0  # No VAT for other categories
    vat = gross * (rate / (1 + rate))
    net = gross - vat
    return net, vat

# ---------------- JOURNAL ENTRY ---------------- #
def build_real_journal_entry(row, income_tax_rate=0.0):
    entries = []
    gross = row['Balance Change']
    net = row['Net Amount']
    vat = row['VAT Amount']
    cat = row['Main Category']
    currency = row.get('Currency', '$')

    if cat == "Revenue":
        # Revenue entry
        entries.append({"Debit": f"Accounts Receivable / Cash ({currency})",
                        "Credit": f"Revenue ({currency})", "Amount": net})
        if vat > 0:
            entries.append({"Debit": f"Accounts Receivable / Cash ({currency})",
                            "Credit": f"VAT Payable ({currency})", "Amount": vat})
        if income_tax_rate > 0:
            tax_amount = net * income_tax_rate
            entries.append({"Debit": f"Income Tax Expense ({currency})",
                            "Credit": f"Income Tax Payable ({currency})", "Amount": tax_amount})

    elif cat == "Expense":
        # Expense entry
        entries.append({"Debit": f"Expense ({currency})",
                        "Credit": f"Accounts Payable / Cash ({currency})", "Amount": net})
        if vat > 0:
            entries.append({"Debit": f"VAT Receivable ({currency})",
                            "Credit": f"Accounts Payable / Cash ({currency})", "Amount": vat})

    elif cat in ["Asset", "Liability", "Equity"]:
        # No VAT applied
        if gross > 0:
            entries.append({"Debit": f"Cash/Bank ({currency})",
                            "Credit": f"{cat} ({currency})", "Amount": gross})
        else:
            entries.append({"Debit": f"{cat} ({currency})",
                            "Credit": f"Cash/Bank ({currency})", "Amount": abs(gross)})

    else:
        # Other categories
        if gross > 0:
            entries.append({"Debit": f"Cash/Bank ({currency})",
                            "Credit": f"Other ({currency})", "Amount": gross})
        else:
            entries.append({"Debit": f"Other ({currency})",
                            "Credit": f"Cash/Bank ({currency})", "Amount": abs(gross)})

    return entries

# ---------------- APPLY VAT & JOURNALS ---------------- #
def apply_vat_and_journals(df, vat_rates, income_tax_rate=0.0):
    net_list, vat_list, journal_list, row_tax_list = [], [], [], []

    for _, row in df.iterrows():
        cat = row['Main Category']
        rate = vat_rates.get(cat, 0)
        gross = row['Balance Change']

        net, vat = extract_vat_from_gross(gross, rate, cat)
        net_list.append(net)
        vat_list.append(vat)

        row2 = row.copy()
        row2['Net Amount'] = net
        row2['VAT Amount'] = vat

        journals = build_real_journal_entry(row2, income_tax_rate)
        journal_list.append(journals)

        # Income tax per row only for revenue
        row_tax_list.append(net * income_tax_rate if cat == "Revenue" else 0)

    df['Net Amount'] = net_list
    df['VAT Amount'] = vat_list
    df['Income Tax per Row'] = row_tax_list
    df['Journal Entries'] = journal_list

    return df

# ---------------- PROFIT & TAX ---------------- #
def compute_profit_and_income_tax(df, tax_rate):
    total_rev = df[df['Main Category']=="Revenue"]['Net Amount'].sum()
    total_exp = df[df['Main Category']=="Expense"]['Net Amount'].sum()
    profit = total_rev - total_exp
    tax = profit * tax_rate if profit > 0 else 0
    return total_rev, total_exp, profit, tax

# ---------------- PDF REPORT ---------------- #
class ERPReportPDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "ERP Accounting Summary Report", ln=True, align="C")
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def add_kpis(self, total_gross, total_net, total_vat, total_rev, total_exp, profit, total_income_tax, currency="$"):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "💰 Key Financial Metrics", ln=True)
        self.set_font("Arial", "", 11)
        metrics = [
            ("Total Gross", total_gross),
            ("Total Net", total_net),
            ("Total VAT", total_vat),
            ("Revenue (Net)", total_rev),
            ("Expense (Net)", total_exp),
            ("Profit", profit),
            ("Income Tax", total_income_tax)
        ]
        for name, val in metrics:
            self.cell(0, 7, f"{name}: {currency}{val:,.2f}", ln=True)
        self.ln(5)

    def add_summary_table(self, df_summary):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "📊 Summary by Main Category", ln=True)
        self.set_font("Arial", "", 10)
        col_widths = [60, 40, 40, 40]
        self.set_fill_color(200, 220, 255)
        headers = ["Category","Net Amount","VAT Amount","Income Tax"]
        for i,h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True)
        self.ln()
        for _, row in df_summary.iterrows():
            currency = row.get('Currency', '$')
            self.cell(col_widths[0], 6, str(row['Main Category']), border=1)
            self.cell(col_widths[1], 6, f"{currency}{row['Net Amount']:,.2f}", border=1)
            self.cell(col_widths[2], 6, f"{currency}{row['VAT Amount']:,.2f}", border=1)
            self.cell(col_widths[3], 6, f"{currency}{row['Income Tax per Row']:,.2f}", border=1)
            self.ln()
        self.ln(5)

    def add_tax_table(self, total_rev, total_exp, profit, total_income_tax, currency="$"):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "🧾 Tax Calculation Table", ln=True)
        self.set_font("Arial", "", 10)
        col_widths = [60, 60]
        headers = ["Metric", "Amount"]
        for i,h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True)
        self.ln()
        rows = [
            ("Revenue (Net)", total_rev),
            ("Expense (Net)", total_exp),
            ("Profit", profit),
            ("Income Tax", total_income_tax)
        ]
        for name, val in rows:
            self.cell(col_widths[0], 6, name, border=1)
            self.cell(col_widths[1], 6, f"{currency}{val:,.2f}", border=1)
            self.ln()
        self.ln(5)

    def add_journals_appendix(self, df_taxed):
        self.add_page()
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "📚 Important Journal Entries Appendix", ln=True)
        self.set_font("Arial", "", 10)
        for i, row in df_taxed.iterrows():
            if row['VAT Amount']==0 and row['Income Tax per Row']==0:
                continue
            currency = row.get('Currency', '$')
            self.set_font("Arial", "B", 10)
            self.cell(0, 6, f"Transaction {i+1}: {row['Description']}", ln=True)
            self.set_font("Arial", "", 9)
            self.cell(0, 6, f"Category: {row['Main Category']} — Gross: {currency}{row['Balance Change']:,.2f} — Net: {currency}{row['Net Amount']:,.2f} — VAT: {currency}{row['VAT Amount']:,.2f} — Income Tax: {currency}{row['Income Tax per Row']:,.2f}", ln=True)
            col_widths = [60, 60, 60]
            self.set_font("Arial", "B", 9)
            self.cell(col_widths[0], 5, "Debit", border=1, fill=True)
            self.cell(col_widths[1], 5, "Credit", border=1, fill=True)
            self.cell(col_widths[2], 5, "Amount", border=1, fill=True)
            self.ln()
            self.set_font("Arial", "", 9)
            for j in row['Journal Entries']:
                if j['Amount']==0:
                    continue
                self.cell(col_widths[0], 5, j['Debit'], border=1)
                self.cell(col_widths[1], 5, j['Credit'], border=1)
                self.cell(col_widths[2], 5, f"{currency}{j['Amount']:,.2f}", border=1)
                self.ln()
            self.ln(3)
            if self.get_y() > 260:
                self.add_page()

# ---------------- STREAMLIT APP ---------------- #
def tax_module():
    st.title("Advanced ERP Accounting Summary (Multi-Currency)")

    df = load_data()
    if df.empty:
        return

    df = preprocess_amounts(df)

    # ---------------- FILTERS ---------------- #
    st.sidebar.header("Filters")
    start_date = st.sidebar.date_input("Start Date", df['Date'].min())
    end_date = st.sidebar.date_input("End Date", df['Date'].max())
    main_categories = st.sidebar.multiselect("Main Category", options=sorted(df['Main Category'].unique()), default=sorted(df['Main Category'].unique()))
    sub_categories = st.sidebar.multiselect("Subcategory", options=sorted(df['Subcategory'].unique()), default=sorted(df['Subcategory'].unique()))
    
    # --- Currency select box --- #
    selected_currency = st.sidebar.selectbox("Select Currency", options=sorted(df['Currency'].unique()), index=0)

    df_filtered = df[
        (df['Date'] >= pd.to_datetime(start_date)) &
        (df['Date'] <= pd.to_datetime(end_date)) &
        (df['Main Category'].isin(main_categories)) &
        (df['Subcategory'].isin(sub_categories)) &
        (df['Currency'] == selected_currency)
    ]

    if df_filtered.empty:
        st.info("No transactions found for selected filters.")
        return

    # VAT Rates
    st.subheader("VAT Rates")
    vat_rates = {cat: st.number_input(f"VAT Rate for {cat} (%)", min_value=0.0, max_value=100.0, value=10.0)/100 for cat in df_filtered['Main Category'].unique()}

    # Income Tax
    st.subheader("Income Tax Rate")
    income_tax_rate = st.number_input("Income Tax Rate (%)", min_value=0.0, max_value=100.0, value=20.0)/100

    # Apply VAT & Journals
    df_taxed = apply_vat_and_journals(df_filtered, vat_rates, income_tax_rate)

    # KPIs
    total_vat = df_taxed['VAT Amount'].sum()
    total_net = df_taxed['Net Amount'].sum()
    total_gross = df_taxed['Balance Change'].sum()
    total_rev, total_exp, profit, total_income_tax = compute_profit_and_income_tax(df_taxed, income_tax_rate)

    # Display KPIs
    st.subheader("VAT & Profit Summary")
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Total Gross", f"{selected_currency}{total_gross:,.2f}")
    colB.metric("Total Net", f"{selected_currency}{total_net:,.2f}")
    colC.metric("Total VAT", f"{selected_currency}{total_vat:,.2f}")
    colD.metric("Profit", f"{selected_currency}{profit:,.2f}")

    # Summary Table
    st.subheader("Summary by Main Category")
    summary_df = df_taxed.groupby(["Main Category"])[["Net Amount","VAT Amount","Income Tax per Row"]].sum().reset_index()
    summary_df['Currency'] = selected_currency
    st.dataframe(summary_df)

    # Plot
    fig = px.bar(summary_df, x="Main Category", y=["Net Amount","VAT Amount","Income Tax per Row"],
                 color='Currency', text_auto=True, title="Category Summary", barmode="group", height=450)
    st.plotly_chart(fig, use_container_width=True)

    # PDF download
    if st.button("⬇️ Download PDF Report"):
        pdf = ERPReportPDF()
        pdf.add_page()
        pdf.add_kpis(total_gross,total_net,total_vat,total_rev,total_exp,profit,total_income_tax,currency=selected_currency)
        pdf.add_summary_table(summary_df)
        pdf.add_tax_table(total_rev,total_exp,profit,total_income_tax,currency=selected_currency)
        pdf.add_journals_appendix(df_taxed)

        # Add chart as image
        fig_bytes = pio.to_image(fig, format='png', width=800, height=500, scale=2)
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "📊 Category Summary Chart", ln=True)
        pdf.image(io.BytesIO(fig_bytes), x=10, y=25, w=pdf.w - 20)

        pdf_bytes = pdf.output(dest='S').encode('latin1')
        st.download_button("Download PDF", data=pdf_bytes, file_name="ERP_Full_Report.pdf", mime="application/pdf")

# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    tax_module()
