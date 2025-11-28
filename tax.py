"""
tax.py

Accounting-grade tax & VAT module (Streamlit-compatible).

Features:
- Load transactions from st.session_state['processed_df'] (defensive)
- Per-category VAT rates (user input)
- Per-row journal entries (Revenue/Expense/Asset/Liability/Equity handling)
- Income tax: per-row estimate (if requested) and aggregated profit-level tax
- PDF export with Unicode-safe TTF font (DejaVuSans recommended)
- Plotly chart embedding into PDF via kaleido
- Clean, modular helper functions

Drop into your app and call tax_module() from Streamlit.
"""

from __future__ import annotations
import os
import io
import re
import tempfile
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import plotly.express as px
import plotly.io as pio
from fpdf import FPDF

# If using in Streamlit:
try:
    import streamlit as st
except Exception:
    st = None  # Module still works for offline usage; streamlit-specific calls are guarded

# -----------------------------
# CONFIG / CONSTANTS
# -----------------------------
FONT_PATH = os.path.join("fonts", "DejaVuSans.ttf")  # recommended: put DejaVuSans.ttf there
DEFAULT_FONT = "DejaVu"
DEFAULT_CURRENCY_SYMBOL = "$"
CURRENCY_MAP = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR", "¥": "JPY"}
CURRENCY_SYMBOLS = list(CURRENCY_MAP.keys())

# -----------------------------
# Helper utilities
# -----------------------------
def _clean_text_for_pdf(text: Any) -> str:
    """
    Replace characters not supported by latin-1 and common long dashes,
    returning a safe-as-latin1 string. Prefer to use a Unicode TTF font.
    """
    if text is None:
        return ""
    s = str(text)
    # replace common unicode punctuation with ascii equivalents
    replacements = {
        "\u2014": "-",  # em dash
        "\u2013": "-",  # en dash
        "\u2022": "*",  # bullet
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    # optionally remove other problematic characters outside latin-1
    try:
        s.encode("latin-1")
        return s
    except UnicodeEncodeError:
        # fallback: replace non-latin1 chars
        safe = s.encode("latin-1", errors="replace").decode("latin-1")
        return safe

def _extract_currency_symbol(value: Any) -> str:
    """Return the first currency symbol found (fallback to DEFAULT_CURRENCY_SYMBOL)."""
    if value is None:
        return DEFAULT_CURRENCY_SYMBOL
    s = str(value)
    for sym in CURRENCY_SYMBOLS:
        if sym in s:
            return sym
    return DEFAULT_CURRENCY_SYMBOL

def _clean_numeric(value: Any) -> float:
    """Keep digits, dot and minus; return float or 0.0."""
    if value is None:
        return 0.0
    s = str(value)
    s = re.sub(r"[^\d\.\-]", "", s)
    try:
        return float(s) if s not in ("", ".", "-") else 0.0
    except Exception:
        return 0.0

# -----------------------------
# Data loading / preprocessing
# -----------------------------
def load_transactions_from_session() -> pd.DataFrame:
    """
    Load processed_df from streamlit session state. Defensive defaults.
    If streamlit isn't present or session key is missing, returns empty DataFrame.
    """
    if st is None:
        raise RuntimeError("Streamlit is required for load_transactions_from_session() in this build.")
    if "processed_df" not in st.session_state:
        st.warning("No processed_df found in st.session_state. Upload first.")
        return pd.DataFrame()
    df = st.session_state["processed_df"].copy()
    # allow common CSV column names
    if "Amount" in df.columns and "Balance Change" not in df.columns:
        df["Balance Change"] = df["Amount"]
    # ensure date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").fillna(pd.Timestamp.today())
    else:
        df["Date"] = pd.Timestamp.today()
    # ensure columns
    for c in ["Main Category", "Subcategory", "Balance Change", "Description"]:
        if c not in df.columns:
            df[c] = "Unknown" if c != "Balance Change" else 0.0
    return df

def preprocess_amounts(df: pd.DataFrame) -> pd.DataFrame:
    """Extract currency symbol and clean numeric Balance Change into float."""
    df = df.copy()
    df["Currency"] = df["Balance Change"].apply(_extract_currency_symbol)
    df["Balance Change"] = df["Balance Change"].apply(_clean_numeric)
    return df

# -----------------------------
# VAT / TAX core logic
# -----------------------------
def extract_vat_from_gross(gross: float, rate: float, category: str) -> Tuple[float, float]:
    """
    Given gross (amount including VAT) and rate (decimal, e.g., 0.10),
    return (net_amount, vat_amount). Only applies for categories 'Revenue' or 'Expense'.
    Returns signed values preserving the sign of gross.
    """
    if rate is None or rate == 0 or str(category).strip() == "":
        return float(gross), 0.0
    cat_key = str(category).strip().lower()
    if cat_key not in ("revenue", "expense"):
        return float(gross), 0.0
    sign = 1.0 if gross >= 0 else -1.0
    vat = abs(gross) * (rate / (1.0 + rate)) * sign
    net = gross - vat
    # round sensibly to cents
    return round(net, 8), round(vat, 8)

def build_row_journals(row: pd.Series, income_tax_rate: float = 0.0) -> List[Dict[str, Any]]:
    """
    Build per-row journal entries following double-entry:
     - Revenue: Debit Cash/AR (gross), Credit Revenue (net), Credit VAT Payable (vat)
     - Expense: Debit Expense (net), Debit VAT Receivable (vat), Credit Cash/AP (gross)
     - Asset/Liability/Equity/Other: simple cash movement entry
    Additionally, if income_tax_rate>0, optionally add per-row income tax estimate (debit expense / credit payable).
    Returns list of small dictionaries describing debit/credit/amount.
    """
    journals: List[Dict[str, Any]] = []
    cat = str(row.get("Main Category", "")).strip().lower()
    cur_sym = row.get("Currency", DEFAULT_CURRENCY_SYMBOL) or DEFAULT_CURRENCY_SYMBOL
    net = float(row.get("Net Amount", 0.0))
    vat = float(row.get("VAT Amount", 0.0))
    gross = float(row.get("Balance Change", 0.0))

    # Revenue
    if cat == "revenue":
        if gross != 0:
            journals.append({"Debit": f"Accounts Receivable / Cash ({cur_sym})", "Credit": f"Revenue ({cur_sym})", "Amount": round(abs(net), 2)})
        if abs(vat) > 0:
            journals.append({"Debit": f"Accounts Receivable / Cash ({cur_sym})", "Credit": f"VAT Payable ({cur_sym})", "Amount": round(abs(vat), 2)})
        # per-row income tax estimate entry (your system may or may not want it)
        if income_tax_rate and income_tax_rate > 0 and abs(net) > 0:
            tax_amt = abs(net) * income_tax_rate
            journals.append({"Debit": f"Income Tax Expense ({cur_sym})", "Credit": f"Income Tax Payable ({cur_sym})", "Amount": round(tax_amt, 2)})
    # Expense
    elif cat == "expense":
        if abs(net) > 0:
            journals.append({"Debit": f"Expense ({cur_sym})", "Credit": f"Accounts Payable / Cash ({cur_sym})", "Amount": round(abs(net), 2)})
        if abs(vat) > 0:
            journals.append({"Debit": f"VAT Receivable ({cur_sym})", "Credit": f"Accounts Payable / Cash ({cur_sym})", "Amount": round(abs(vat), 2)})
    # Asset / liability / equity
    elif cat in ("asset", "liability", "equity"):
        if gross >= 0:
            journals.append({"Debit": f"Cash/Bank ({cur_sym})", "Credit": f"{row.get('Main Category')} ({cur_sym})", "Amount": round(abs(gross), 2)})
        else:
            journals.append({"Debit": f"{row.get('Main Category')} ({cur_sym})", "Credit": f"Cash/Bank ({cur_sym})", "Amount": round(abs(gross), 2)})
    else:
        # default cash movement
        if gross >= 0:
            journals.append({"Debit": f"Cash/Bank ({cur_sym})", "Credit": f"Other ({cur_sym})", "Amount": round(abs(gross), 2)})
        else:
            journals.append({"Debit": f"Other ({cur_sym})", "Credit": f"Cash/Bank ({cur_sym})", "Amount": round(abs(gross), 2)})

    return journals

def apply_vat_and_journals(df: pd.DataFrame, vat_rates_by_category: Dict[str, float], income_tax_rate: float = 0.0) -> pd.DataFrame:
    """
    Apply VAT extraction and produce per-row journals.
    vat_rates_by_category: dict mapping Main Category -> decimal rate (e.g., 0.10)
    income_tax_rate: decimal (0.20 for 20%) used for per-row estimate (kept separate from final profit-level tax)
    Returns a copy of df with added columns: Net Amount, VAT Amount, Journal Entries, Income Tax per Row.
    """
    df = df.copy()
    net_list, vat_list, journals_list, per_row_tax = [], [], [], []

    for _, row in df.iterrows():
        cat = row.get("Main Category", "")
        # Normalize lookup: try exact category and lower-case fallback
        rate = 0.0
        if isinstance(vat_rates_by_category, dict):
            rate = vat_rates_by_category.get(cat, vat_rates_by_category.get(str(cat).lower(), 0.0))
        gross = float(row.get("Balance Change", 0.0))
        net_amt, vat_amt = extract_vat_from_gross(gross, rate, cat)
        net_list.append(net_amt)
        vat_list.append(vat_amt)

        row_copy = row.copy()
        row_copy["Net Amount"] = net_amt
        row_copy["VAT Amount"] = vat_amt

        journals = build_row_journals(row_copy, income_tax_rate)
        journals_list.append(journals)

        per_row_tax.append(round(abs(net_amt) * income_tax_rate, 8) if str(cat).strip().lower() == "revenue" else 0.0)

    df["Net Amount"] = net_list
    df["VAT Amount"] = vat_list
    df["Journal Entries"] = journals_list
    df["Income Tax per Row"] = per_row_tax
    return df

def aggregate_vat(df: pd.DataFrame) -> Tuple[float, float, float]:
    """Return (total_vat_collected, total_vat_paid, net_vat). VAT amounts are considered absolute where appropriate."""
    df = df.copy()
    rev_vat = df[df["Main Category"].str.lower() == "revenue"]["VAT Amount"].sum()
    exp_vat = df[df["Main Category"].str.lower() == "expense"]["VAT Amount"].sum()
    total_vat_collected = df[df["Main Category"].str.lower() == "revenue"]["VAT Amount"].apply(abs).sum()
    total_vat_paid = df[df["Main Category"].str.lower() == "expense"]["VAT Amount"].apply(abs).sum()
    net_vat = total_vat_collected - total_vat_paid
    return total_vat_collected, total_vat_paid, net_vat

def compute_profit_and_income_tax(df: pd.DataFrame, income_tax_rate: float) -> Tuple[float, float, float, float, float]:
    """
    Compute totals:
    - total_rev_net: sum of revenue net amounts
    - total_exp_net: sum of expense net amounts (expected positive)
    - profit_before_tax
    - income_tax_on_profit (applied only if profit positive)
    - profit_after_tax
    """
    total_rev_net = df[df["Main Category"].str.lower() == "revenue"]["Net Amount"].sum()
    total_exp_net = df[df["Main Category"].str.lower() == "expense"]["Net Amount"].sum()
    profit_before_tax = total_rev_net - total_exp_net
    income_tax = profit_before_tax * income_tax_rate if profit_before_tax > 0 else 0.0
    profit_after_tax = profit_before_tax - income_tax
    return total_rev_net, total_exp_net, profit_before_tax, income_tax, profit_after_tax

# -----------------------------
# PDF Report (Unicode-safe)
# -----------------------------
class ERPReportPDF(FPDF):
    def __init__(self, font_path: Optional[str] = None):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=12)
        # load a unicode-capable TTF if possible
        ttf = font_path or FONT_PATH
        if os.path.exists(ttf):
            try:
                self.add_font(DEFAULT_FONT, "", ttf, uni=True)
                self._pdf_font = DEFAULT_FONT
            except Exception:
                self._pdf_font = "Helvetica"
        else:
            # fallback, but we'll sanitize text before writing
            self._pdf_font = "Helvetica"
        self.set_font(self._pdf_font, size=11)

    def header(self):
        self.set_font(self._pdf_font, style="B", size=14)
        self.cell(0, 10, _clean_text_for_pdf("ERP Accounting — Tax Summary"), ln=True, align="C")
        self.ln(4)
        self.set_font(self._pdf_font, size=11)

    def footer(self):
        self.set_y(-12)
        self.set_font(self._pdf_font, style="I", size=9)
        self.cell(0, 10, _clean_text_for_pdf(f"Page {self.page_no()}"), align="C")

    def add_kpis(self, total_gross, total_net, total_vat_collected, total_vat_paid, net_vat,
                 total_rev_net, total_exp_net, profit_bt, income_tax, profit_at, currency_sym: str = DEFAULT_CURRENCY_SYMBOL):
        self.set_font(self._pdf_font, style="B", size=12)
        self.cell(0, 8, _clean_text_for_pdf("Key Metrics"), ln=True)
        self.set_font(self._pdf_font, size=10)
        rows = [
            ("Total Gross (incl. VAT)", total_gross),
            ("Total Net (sum Net Amount)", total_net),
            ("VAT Collected (Sales)", total_vat_collected),
            ("VAT Paid (Purchases)", total_vat_paid),
            ("Net VAT (Collected - Paid)", net_vat),
            ("Revenue (Net)", total_rev_net),
            ("Expense (Net)", total_exp_net),
            ("Profit Before Tax", profit_bt),
            ("Income Tax (on PBT)", income_tax),
            ("Profit After Tax", profit_at),
        ]
        for label, val in rows:
            self.cell(0, 6, _clean_text_for_pdf(f"{label}: {format_currency(val, currency_sym)}"), ln=True)
        self.ln(4)

    def add_summary_table(self, df_summary: pd.DataFrame):
        self.set_font(self._pdf_font, style="B", size=12)
        self.cell(0, 8, _clean_text_for_pdf("Summary by Main Category"), ln=True)
        self.set_font(self._pdf_font, size=10)
        cols = ["Main Category", "Net Amount", "VAT Amount", "Income Tax per Row"]
        widths = [60, 40, 40, 40]
        # header
        for i, h in enumerate(cols):
            self.cell(widths[i], 7, _clean_text_for_pdf(h), border=1, fill=True)
        self.ln()
        for _, r in df_summary.iterrows():
            cur = r.get("Currency", DEFAULT_CURRENCY_SYMBOL)
            self.cell(widths[0], 6, _clean_text_for_pdf(str(r.get("Main Category", "")))[:30], border=1)
            self.cell(widths[1], 6, _clean_text_for_pdf(format_currency(r.get("Net Amount", 0.0), cur)), border=1)
            self.cell(widths[2], 6, _clean_text_for_pdf(format_currency(r.get("VAT Amount", 0.0), cur)), border=1)
            self.cell(widths[3], 6, _clean_text_for_pdf(format_currency(r.get("Income Tax per Row", 0.0), cur)), border=1)
            self.ln()
        self.ln(6)

    def add_journals(self, df_taxed: pd.DataFrame):
        self.add_page()
        self.set_font(self._pdf_font, style="B", size=12)
        self.cell(0, 8, _clean_text_for_pdf("Journal Entries (Per Transaction)"), ln=True)
        self.ln(2)
        self.set_font(self._pdf_font, size=9)
        for idx, r in df_taxed.iterrows():
            journals = r.get("Journal Entries", []) or []
            if not journals:
                continue
            self.set_font(self._pdf_font, style="B", size=10)
            self.cell(0, 6, _clean_text_for_pdf(f"Transaction {idx+1}: {r.get('Description','')[:60]}"), ln=True)
            self.set_font(self._pdf_font, size=9)
            for j in journals:
                self.cell(0, 5, _clean_text_for_pdf(f"Debit: {j.get('Debit','')}  |  Credit: {j.get('Credit','')}  |  Amount: {format_currency(j.get('Amount',0.0), r.get('Currency', DEFAULT_CURRENCY_SYMBOL))}"), ln=True)
            self.ln(2)
            if self.get_y() > 260:
                self.add_page()

    def add_chart(self, fig):
        """
        Embed a plotly figure by converting to PNG (requires kaleido).
        Creates a temporary file and removes it afterward.
        """
        self.add_page()
        self.set_font(self._pdf_font, style="B", size=12)
        self.cell(0, 8, _clean_text_for_pdf("Category Summary Chart"), ln=True)
        try:
            fig_bytes = pio.to_image(fig, format="png", width=900, height=500, scale=2)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(fig_bytes)
                tmp_path = tmp.name
            self.image(tmp_path, x=10, y=25, w=self.w - 20)
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        except Exception as e:
            self.set_font(self._pdf_font, style="I", size=10)
            self.cell(0, 8, _clean_text_for_pdf(f"Chart could not be rendered: {e}"), ln=True)

# -----------------------------
# Format helpers
# -----------------------------
def format_currency(amount: float, symbol: str = DEFAULT_CURRENCY_SYMBOL) -> str:
    try:
        amount = float(amount)
        if amount < 0:
            return f"-{symbol}{abs(amount):,.2f}"
        return f"{symbol}{amount:,.2f}"
    except Exception:
        return f"{symbol}{amount}"

# -----------------------------
# Streamlit-facing tax module
# -----------------------------
def tax_module():
    """Streamlit page function. Expects st.session_state['processed_df'] present."""
    if st is None:
        raise RuntimeError("Streamlit is required to run tax_module()")

    st.set_page_config(page_title="Tax & VAT Module", layout="wide")
    st.title("Accounting-Grade VAT & Income Tax Module")

    df = load_transactions_from_session()
    if df.empty:
        return
    df = preprocess_amounts(df)

    # Filters
    st.sidebar.header("Filters")
    start_date = st.sidebar.date_input("Start Date", df["Date"].min().date())
    end_date = st.sidebar.date_input("End Date", df["Date"].max().date())
    main_cats = st.sidebar.multiselect("Main Category", options=sorted(df["Main Category"].unique()), default=sorted(df["Main Category"].unique()))
    subs = st.sidebar.multiselect("Subcategory", options=sorted(df["Subcategory"].unique()), default=sorted(df["Subcategory"].unique()))
    currency_options = sorted(df["Currency"].unique())
    selected_currency = st.sidebar.selectbox("Currency Symbol", options=currency_options)

    df_f = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]
    if main_cats:
        df_f = df_f[df_f["Main Category"].isin(main_cats)]
    if subs:
        df_f = df_f[df_f["Subcategory"].isin(subs)]
    if selected_currency:
        df_f = df_f[df_f["Currency"] == selected_currency]

    if df_f.empty:
        st.info("No transactions for selected filters.")
        return

    # VAT & income tax inputs
    st.sidebar.header("Tax Settings")
    vat_rates = {}
    for cat in sorted(df_f["Main Category"].unique()):
        vat_rates[cat] = st.sidebar.number_input(f"VAT % for {cat}", min_value=0.0, max_value=100.0, value=0.0) / 100.0
    income_tax_rate = st.sidebar.number_input("Income Tax % (applies to profit)", min_value=0.0, max_value=100.0, value=20.0) / 100.0
    # Option: apply per-row income tax postings
    per_row_tax_posting = st.sidebar.checkbox("Create per-row income-tax journal entries (debit expense/credit payable)", value=False)

    # Apply VAT & journals
    df_taxed = apply_vat_and_journals(df_f, vat_rates, income_tax_rate if per_row_tax_posting else 0.0)

    total_vat_collected, total_vat_paid, net_vat = aggregate_vat(df_taxed)
    total_net = df_taxed["Net Amount"].sum()
    total_gross = df_taxed["Balance Change"].sum()
    total_rev_net, total_exp_net, profit_bt, income_tax_on_profit, profit_at = compute_profit_and_income_tax(df_taxed, income_tax_rate)

    # KPIs
    st.subheader("VAT & Profit Summary")
    a, b, c, d = st.columns(4)
    a.metric("Total Gross", format_currency(total_gross, selected_currency))
    b.metric("Total Net", format_currency(total_net, selected_currency))
    c.metric("Total VAT (collected - if any)", format_currency(total_vat_collected, selected_currency))
    d.metric("Profit (Before Tax)", format_currency(profit_bt, selected_currency))
    st.metric("Profit After Tax (on profit)", format_currency(profit_at, selected_currency))

    # Summary table
    summary_df = df_taxed.groupby("Main Category")[["Net Amount", "VAT Amount", "Income Tax per Row"]].sum().reset_index()
    summary_df["Currency"] = selected_currency
    st.subheader("Summary by Main Category")
    st.dataframe(summary_df)

    # Chart
    fig = px.bar(
    summary_df,
    x="Main Category",
    y=["Net Amount", "VAT Amount", "Income Tax per Row"],
    barmode="group",
    title="Category Summary",
    height=450,
    color_discrete_map={
        "Net Amount": "#1f77b4",             # Blue
        "VAT Amount": "#ff7f0e",             # Orange
        "Income Tax per Row": "#2ca02c"      # Green
    }
)

    st.plotly_chart(fig, use_container_width=True)

    # PDF generation
    if st.button("⬇️ Generate PDF Report"):
        pdf = ERPReportPDF(font_path=FONT_PATH if os.path.exists(FONT_PATH) else None)
        pdf.add_page()
        pdf.add_kpis(total_gross, total_net, total_vat_collected, total_vat_paid, net_vat,
                     total_rev_net, total_exp_net, profit_bt, income_tax_on_profit, profit_at,
                     currency_sym=selected_currency)
        pdf.add_summary_table(summary_df)
        pdf.add_journals(df_taxed)
        pdf.add_chart(fig)

        pdf_bytes = bytes(pdf.output(dest="S"))  # convert bytearray -> bytes

        st.download_button("Download PDF", data=pdf_bytes, file_name="Tax_Report.pdf", mime="application/pdf")

