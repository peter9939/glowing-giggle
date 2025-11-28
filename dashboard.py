# dashboard_module_full_export.py
import logging
from io import BytesIO
from typing import Dict, Optional
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Constants ----------------
REQUIRED_COLS = ['Date', 'Main Category', 'Subcategory', 'Balance Change', 'Description', 'Auto-Matched']
CASH_ACCOUNT = "Cash / Bank"
CATEGORY_DC_MAP = {
    'revenue': ('Credit', 'Debit'),
    'expense': ('Debit', 'Credit'),
    'asset': ('Debit', 'Credit'),
    'liability': ('Credit', 'Debit'),
    'equity': ('Credit', 'Debit')
}
CHART_COLORS = px.colors.qualitative.Safe
CURRENCY_MAP = {'$': 'USD', '€': 'EUR', '£': 'GBP', '₹': 'INR', '¥': 'JPY'}

# ---------------- Utilities ----------------
def preprocess_amounts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Try to detect currency symbol or code in Balance Change; default to $
    df['Currency'] = df.get('Balance Change', '').astype(str).str.extract(r'([A-Z$€£₹¥]+)')[0].fillna('$')
    df['Currency'] = df['Currency'].map(CURRENCY_MAP).fillna('USD')
    df['Balance Change'] = df['Balance Change'].astype(str).str.replace(r"[^\d\.-]", "", regex=True)
    df['Balance Change'] = pd.to_numeric(df['Balance Change'], errors='coerce').fillna(0.0)
    return df

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in REQUIRED_COLS:
        if c not in df.columns:
            logger.info("Adding missing column: %s", c)
            if c == 'Date':
                df[c] = pd.Timestamp.today()
            elif c == 'Balance Change':
                df[c] = 0.0
            elif c == 'Auto-Matched':
                df[c] = False
            else:
                df[c] = "Unknown"
    # Normalize types
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').fillna(pd.Timestamp.today())
    df['Balance Change'] = pd.to_numeric(df['Balance Change'], errors='coerce').fillna(0.0)
    df['Main Category'] = df['Main Category'].astype(str)
    df['Subcategory'] = df['Subcategory'].astype(str)
    df['Description'] = df['Description'].astype(str)
    return df

def format_currency(x, symbol='$'):
    try:
        return f"{symbol}{x:,.2f}"
    except Exception:
        return str(x)

# ---------------- KPI Calculations ----------------
def compute_kpis(df: pd.DataFrame) -> Dict[str, float]:
    lookup = lambda key: df[df['Main Category'].str.lower() == key]['Balance Change'].sum()
    revenue = lookup('revenue')
    expense = lookup('expense')
    net_profit = revenue - expense
    assets = lookup('asset')
    liabilities = lookup('liability')
    equity = lookup('equity')
    current_ratio = (assets / liabilities) if liabilities not in (0, 0.0) else float('inf')
    return {
        'Revenue': revenue,
        'Expense': expense,
        'Net Profit': net_profit,
        'Total Assets': assets,
        'Total Liabilities': liabilities,
        'Total Equity': equity,
        'Current Ratio': current_ratio
    }

# ---------------- Matplotlib image helpers (stable; no Kaleido) ----------------
def _png_from_figure_bytes(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def bar_image_from_df(x, y, title: str = "", xlabel: str = "", ylabel: str = "") -> bytes:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    return _png_from_figure_bytes(fig)

def grouped_bar_image_from_df(index_values, columns_df: pd.DataFrame, title: str = "", xlabel: str = "", ylabel: str = "") -> bytes:
    fig, ax = plt.subplots(figsize=(10, 5))
    # columns_df: DataFrame with columns to plot (numeric)
    n = len(columns_df.columns) if hasattr(columns_df, 'columns') else 1
    ind = np.arange(len(index_values))
    width = 0.8 / max(1, n)
    for i, col in enumerate(columns_df.columns):
        ax.bar(ind + i * width, columns_df[col].values, width=width, label=col)
    ax.set_xticks(ind + width * (n - 1) / 2)
    ax.set_xticklabels(index_values, rotation=45, ha='right')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if n > 1:
        ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    return _png_from_figure_bytes(fig)

def line_image_from_df(x, y, title: str = "", xlabel: str = "", ylabel: str = "") -> bytes:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, marker='o')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    # If x are dates, format nicely
    try:
        if np.issubdtype(np.array(x).dtype, np.datetime64):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    except Exception:
        pass
    return _png_from_figure_bytes(fig)

# ---------------- Chart & KPI Download Helpers ----------------
def download_matplotlib_image(png_bytes: bytes, filename: str, label: str):
    st.download_button(
        label=label,
        data=png_bytes,
        file_name=filename,
        mime="image/png"
    )

def download_kpi_image(kpi_dict: Dict[str, float], filename="kpi.png", width=800, height=300) -> Image.Image:
    # Draw a simple KPI image and offer download; return PIL.Image for embedding in Excel
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)
    font_size = 18
    try:
        # Try Arial; fallback to default
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    x, y = 20, 20
    draw.text((x, y), "Key Performance Indicators", fill="black", font=font)
    y += font_size + 10
    for k, v in kpi_dict.items():
        val_str = f"{v:,.2f}" if isinstance(v, (int, float)) and v != float('inf') else str(v)
        draw.text((x, y), f"{k}: {val_str}", fill="black", font=font)
        y += font_size + 6
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    st.download_button(
        label="📥 Download KPI Image",
        data=buf.getvalue(),
        file_name=filename,
        mime="image/png"
    )
    buf.seek(0)
    return Image.open(buf)

# ---------------- Table Download Helper ----------------
def download_df_excel(df: pd.DataFrame, filename="table.xlsx") -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False)
    buffer.seek(0)
    st.download_button("📥 Download Excel", buffer.getvalue(), filename)
    return buffer.getvalue()

# ---------------- Journal & Ledger ----------------
def generate_journal_entries(df: pd.DataFrame) -> pd.DataFrame:
    journal = []
    df = df.copy()
    df['main_lower'] = df['Main Category'].str.lower()
    for _, r in df.iterrows():
        amt = abs(float(r.get('Balance Change', 0) or 0))
        if amt == 0:
            continue
        date = r.get('Date', pd.Timestamp.today())
        desc = r.get('Description', '')
        acct = f"{r.get('Main Category', '')} - {r.get('Subcategory', '')}"
        cat = r.get('main_lower', '')
        if cat in CATEGORY_DC_MAP:
            # CATEGORY_DC_MAP[cat] returns ('Credit' / 'Debit', ...)
            # We interpret it: if first element == 'Credit' then cash is Debit (i.e. Cash / Bank receives debit)
            if CATEGORY_DC_MAP[cat][0] == 'Credit':
                debit_acc, credit_acc = CASH_ACCOUNT, acct
            else:
                debit_acc, credit_acc = acct, CASH_ACCOUNT
            journal.append({'Date': date, 'Description': desc,
                            'Debit Account': debit_acc, 'Debit': amt,
                            'Credit Account': credit_acc, 'Credit': amt})
        else:
            journal.append({'Date': date, 'Description': desc,
                            'Debit Account': acct, 'Debit': amt,
                            'Credit Account': CASH_ACCOUNT, 'Credit': amt})
    return pd.DataFrame(journal)

def build_ledger_from_journal(jdf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in jdf.iterrows():
        rows.append({'Account': r['Debit Account'], 'Date': r['Date'], 'Description': r['Description'],
                     'Debit': r['Debit'], 'Credit': 0.0})
        rows.append({'Account': r['Credit Account'], 'Date': r['Date'], 'Description': r['Description'],
                     'Debit': 0.0, 'Credit': r['Credit']})
    ledger = pd.DataFrame(rows)
    if ledger.empty:
        return ledger
    ledger['Date'] = pd.to_datetime(ledger['Date'], errors='coerce').fillna(pd.Timestamp.today())
    ledger.sort_values(['Account', 'Date'], inplace=True)
    ledger['Running Balance'] = ledger.groupby('Account').apply(lambda g: (g['Debit'] - g['Credit']).cumsum()).reset_index(level=0, drop=True)
    return ledger

# ---------------- Excel Workbook Export with Charts (uses PNG bytes) ----------------
def export_full_excel(kpi: Dict[str, float], tables_dict: Dict[str, pd.DataFrame],
                      charts_images: Dict[str, Optional[bytes]], kpi_img: Image.Image) -> bytes:
    """
    Create an Excel workbook with:
     - KPIs sheet (with KPI table and inserted KPI image)
     - One sheet per table in tables_dict
     - One sheet per chart image in charts_images (if image bytes present)
    Returns the Excel file bytes.
    """
    import xlsxwriter

    def get_unique_name(workbook, base_name: str) -> str:
        name = base_name[:31]
        existing = [ws.name.lower() for ws in workbook.worksheets()]
        if name.lower() not in existing:
            return name
        counter = 1
        while True:
            new_name = f"{name[:28]}({counter})"
            if new_name.lower() not in existing:
                return new_name
            counter += 1

    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book

        # 1) KPI Sheet
        kpi_df = pd.DataFrame(list(kpi.items()), columns=["KPI", "Value"])
        kpi_df.to_excel(writer, sheet_name="KPIs", index=False)
        # Insert KPI image
        try:
            kpi_img_buf = BytesIO()
            kpi_img.save(kpi_img_buf, format="PNG")
            kpi_img_buf.seek(0)
            ws = writer.sheets["KPIs"]
            ws.insert_image("D2", "kpi.png", {'image_data': kpi_img_buf, 'x_scale': 0.8, 'y_scale': 0.8})
        except Exception as e:
            logger.exception("Failed to insert KPI image into Excel: %s", e)

        # 2) Table sheets
        for name, df in tables_dict.items():
            try:
                safe_name = get_unique_name(workbook, name[:31])
                df.to_excel(writer, sheet_name=safe_name, index=False)
            except Exception as e:
                logger.exception("Skipping table %s due to error: %s", name, e)

        # 3) Chart images
        for cname, img_bytes in charts_images.items():
            if not img_bytes:
                # skip None or empty chart images
                continue
            try:
                sheet_name = get_unique_name(workbook, cname[:31])
                ws_chart = workbook.add_worksheet(sheet_name)
                img_buffer = BytesIO(img_bytes)
                ws_chart.insert_image("B2", f"{cname}.png", {'image_data': img_buffer})
            except Exception as e:
                logger.exception("Failed to add chart %s to workbook: %s", cname, e)

        # Exit 'with' to save
    output.seek(0)
    return output.getvalue()

# ---------------- Dashboard UI ----------------
def dashboard_module():
    st.set_page_config(page_title='Accounting Dashboard', layout='wide')
    st.title('📊 Accounting Dashboard (0% VAT — Real Double Entry)')

    # Expecting st.session_state['processed_df'] to be set by an upload/earlier step
    if 'processed_df' not in st.session_state:
        st.error('No processed data found. Please upload data first to session_state["processed_df"].')
        return

    df = st.session_state['processed_df']
    df = preprocess_amounts(df)
    df = ensure_columns(df)

    # Sidebar filters
    st.sidebar.header('Filters')
    currency_options = ['$', '€', '£', '₹', '¥']
    selected_currency = st.sidebar.selectbox('Select Currency', currency_options, index=0)
    min_date = df['Date'].min().date() if not df.empty else pd.Timestamp.today().date()
    max_date = df['Date'].max().date() if not df.empty else pd.Timestamp.today().date()
    start = st.sidebar.date_input('Start Date', min_date)
    end = st.sidebar.date_input('End Date', max_date)
    cats = st.sidebar.multiselect('Main Category', sorted(df['Main Category'].unique()))
    subs = st.sidebar.multiselect('Subcategory', sorted(df['Subcategory'].unique()))
    keyword = st.sidebar.text_input('Search Description')

    df_f = df[(df['Date'] >= pd.to_datetime(start)) & (df['Date'] <= pd.to_datetime(end))]
    if cats:
        df_f = df_f[df_f['Main Category'].isin(cats)]
    if subs:
        df_f = df_f[df_f['Subcategory'].isin(subs)]
    if keyword:
        df_f = df_f[df_f['Description'].str.contains(keyword, case=False, na=False)]

    if df_f.empty:
        st.warning('No data matches filters.')
        return

    # KPIs
    kpi = compute_kpis(df_f)
    st.subheader('Key Performance Indicators')
    c1, c2, c3 = st.columns(3)
    c1.metric('Revenue', format_currency(kpi['Revenue'], selected_currency))
    c2.metric('Expense', format_currency(kpi['Expense'], selected_currency))
    c3.metric('Net Profit', format_currency(kpi['Net Profit'], selected_currency))
    c4, c5, c6 = st.columns(3)
    c4.metric('Total Assets', format_currency(kpi['Total Assets'], selected_currency))
    c5.metric('Total Liabilities', format_currency(kpi['Total Liabilities'], selected_currency))
    c6.metric('Total Equity', format_currency(kpi['Total Equity'], selected_currency))
    ratio_display = f"{kpi['Current Ratio']:.2f}" if kpi['Current Ratio'] != float('inf') else '∞'
    st.metric('Current Ratio', ratio_display)

    kpi_img = download_kpi_image(kpi)

    # Tabs for tables and charts
    tabs = st.tabs(['Trial Balance', 'Income Statement', 'Balance Sheet', 'Cash Flow', 'P&L Monthly', 'Category Drilldown', 'Journal & Ledger'])

    tables_dict: Dict[str, pd.DataFrame] = {}
    charts_dict_bytes: Dict[str, Optional[bytes]] = {}  # store PNG bytes for excel export (None if not available)

    # --- Trial Balance ---
    with tabs[0]:
        st.subheader('Trial Balance')
        tb = df_f.copy()

        def tb_dc(r):
            mc = str(r['Main Category']).lower()
            amt = abs(float(r.get('Balance Change', 0) or 0))
            if mc in ['expense', 'asset']:
                return pd.Series([amt, 0.0])
            if mc in ['revenue', 'liability', 'equity']:
                return pd.Series([0.0, amt])
            return pd.Series([amt, 0.0])

        tb[['Debit', 'Credit']] = tb.apply(tb_dc, axis=1)
        st.dataframe(tb[['Main Category', 'Subcategory', 'Debit', 'Credit']])
        tables_dict['Trial Balance'] = tb[['Main Category', 'Subcategory', 'Debit', 'Credit']]

        # interactive plotly chart
        fig_tb = px.bar(tb, x='Main Category', y='Balance Change', color='Main Category',
                        color_discrete_sequence=CHART_COLORS)
        st.plotly_chart(fig_tb)

        # stable PNG using matplotlib (for download + excel)
        tb_agg = tb.groupby('Main Category')['Balance Change'].sum().reset_index()
        png_tb = bar_image_from_df(tb_agg['Main Category'].astype(str).tolist(), tb_agg['Balance Change'].tolist(),
                                   title="Trial Balance", xlabel="Main Category", ylabel="Amount")
        download_matplotlib_image(png_tb, "trial_balance.png", "📥 Download Trial Balance (PNG)")
        charts_dict_bytes['Trial Balance'] = png_tb

    # --- Income Statement ---
    with tabs[1]:
        st.subheader('Income Statement')
        inc = df_f[df_f['Main Category'].str.lower().isin(['revenue', 'expense'])].copy()
        g = inc.groupby('Main Category')['Balance Change'].sum().reset_index()
        st.dataframe(g)
        tables_dict['Income Statement'] = g
        fig_inc = px.bar(g, x='Main Category', y='Balance Change', color='Main Category', color_discrete_sequence=CHART_COLORS)
        st.plotly_chart(fig_inc)

        png_inc = bar_image_from_df(g['Main Category'].astype(str).tolist(), g['Balance Change'].tolist(),
                                   title="Income Statement", xlabel="Main Category", ylabel="Amount")
        download_matplotlib_image(png_inc, "income_statement.png", "📥 Download Income Statement (PNG)")
        charts_dict_bytes['Income Statement'] = png_inc

    # --- Balance Sheet ---
    with tabs[2]:
        st.subheader('Balance Sheet')
        bs = df_f[df_f['Main Category'].str.lower().isin(['asset', 'liability', 'equity'])].copy()
        bs_sum = bs.groupby('Main Category')['Balance Change'].sum().reset_index()
        st.dataframe(bs_sum)
        tables_dict['Balance Sheet'] = bs_sum
        fig_bs = px.bar(bs_sum, x='Main Category', y='Balance Change', color='Main Category', color_discrete_sequence=CHART_COLORS)
        st.plotly_chart(fig_bs)

        png_bs = bar_image_from_df(bs_sum['Main Category'].astype(str).tolist(), bs_sum['Balance Change'].tolist(),
                                  title="Balance Sheet", xlabel="Main Category", ylabel="Amount")
        download_matplotlib_image(png_bs, "balance_sheet.png", "📥 Download Balance Sheet (PNG)")
        charts_dict_bytes['Balance Sheet'] = png_bs

    # --- Cash Flow ---
    with tabs[3]:
        st.subheader('Cash Flow')
        cf = df_f.copy()
        cf['Month'] = cf['Date'].dt.to_period('M').astype(str)
        cf['Type'] = cf['Main Category'].apply(
            lambda mc: 'Cash In' if mc.lower() == 'revenue' else (
                'Cash Out' if mc.lower() == 'expense' else (
                    'Financing' if mc.lower() in ['equity', 'liability'] else 'Other')))
        g = cf.groupby(['Month', 'Type'])['Balance Change'].sum().reset_index()
        pivot = g.pivot(index='Month', columns='Type', values='Balance Change').fillna(0).reset_index()
        st.dataframe(pivot)
        tables_dict['Cash Flow'] = pivot

        # interactive grouped plotly chart
        if pivot.shape[0] > 0 and pivot.shape[1] > 1:
            value_cols = pivot.columns.tolist()[1:]
            fig_cf = px.bar(pivot, x='Month', y=value_cols, barmode='group', color_discrete_sequence=CHART_COLORS)
            st.plotly_chart(fig_cf)

            # matplotlib grouped bar for stable PNG
            png_cf = grouped_bar_image_from_df(pivot['Month'].astype(str).tolist(), pivot[value_cols],
                                               title="Cash Flow (Monthly)", xlabel="Month", ylabel="Amount")
            download_matplotlib_image(png_cf, "cash_flow.png", "📥 Download Cash Flow (PNG)")
            charts_dict_bytes['Cash Flow'] = png_cf
        else:
            st.info("Not enough data to render Cash Flow chart.")
            charts_dict_bytes['Cash Flow'] = None

    # --- P&L Monthly ---
    with tabs[4]:
        st.subheader('Profit & Loss Monthly')
        pl = df_f[df_f['Main Category'].str.lower().isin(['revenue', 'expense'])].copy()
        if not pl.empty:
            pl['Month'] = pl['Date'].dt.to_period('M')
            g = pl.groupby([pl['Month'].astype(str), 'Main Category'])['Balance Change'].sum().unstack(fill_value=0)
            # Normalize column names to Title case
            g.columns = [c.title() for c in g.columns]
            for col in ['Revenue', 'Expense']:
                if col not in g.columns:
                    g[col] = 0.0
            # Convert index to timestamp for plotting
            g.index = pd.PeriodIndex(g.index, freq='M').to_timestamp()
            g['Net Profit'] = g['Revenue'] - g['Expense']
            st.dataframe(g.reset_index())
            tables_dict['P&L Monthly'] = g.reset_index()

            plot_df = g.reset_index()
            # plot_df['MonthStr'] is string for x-axis
            plot_df['MonthStr'] = plot_df['Month'].dt.to_period('M').astype(str)

            # interactive charts
            fig_pl_bar = px.bar(plot_df, x='MonthStr', y=['Revenue', 'Expense'], barmode='group', color_discrete_sequence=CHART_COLORS)
            st.plotly_chart(fig_pl_bar)
            fig_pl_line = px.line(plot_df, x='MonthStr', y='Net Profit', markers=True, color_discrete_sequence=CHART_COLORS)
            st.plotly_chart(fig_pl_line)

            # stable PNGs via matplotlib
            png_pl_bar = grouped_bar_image_from_df(plot_df['MonthStr'].tolist(), plot_df[['Revenue', 'Expense']],
                                                  title="P&L Monthly (Revenue vs Expense)", xlabel="Month", ylabel="Amount")
            download_matplotlib_image(png_pl_bar, "pl_monthly_bar.png", "📥 Download P&L Monthly (Bar PNG)")
            charts_dict_bytes['P&L Monthly Bar'] = png_pl_bar

            png_pl_line = line_image_from_df(plot_df['MonthStr'].tolist(), plot_df['Net Profit'].tolist(),
                                             title="P&L Monthly - Net Profit", xlabel="Month", ylabel="Net Profit")
            download_matplotlib_image(png_pl_line, "pl_monthly_line.png", "📥 Download P&L Monthly (Line PNG)")
            charts_dict_bytes['P&L Monthly Line'] = png_pl_line
        else:
            st.info("No Revenue/Expense data for the selected range.")
            charts_dict_bytes['P&L Monthly Bar'] = None
            charts_dict_bytes['P&L Monthly Line'] = None

    # --- Category Drilldown ---
    with tabs[5]:
        st.subheader('Category Drilldown')
        drill = df_f.groupby(['Main Category', 'Subcategory'])['Balance Change'].sum().reset_index()
        st.dataframe(drill)
        tables_dict['Category Drilldown'] = drill

        if not drill.empty:
            png_drill = bar_image_from_df(
                drill['Subcategory'].astype(str).tolist(),
                drill['Balance Change'].tolist(),
                title="Category Drilldown",
                xlabel="Subcategory",
                ylabel="Amount"
            )
            download_matplotlib_image(png_drill, "category_drilldown.png", "📥 Download Drilldown (PNG)")
            charts_dict_bytes['Category Drilldown'] = png_drill
        else:
            charts_dict_bytes['Category Drilldown'] = None

    # --- Journal & Ledger ---
    with tabs[6]:
        st.subheader('Journal Entries')
        jdf = generate_journal_entries(df_f)
        st.dataframe(jdf)
        tables_dict['Journal Entries'] = jdf

        st.subheader('Ledger')
        ledger = build_ledger_from_journal(jdf)
        st.dataframe(ledger)
        tables_dict['Ledger'] = ledger

    # --- Full Excel Download ---
    st.subheader("📥 Download Full Excel Workbook")
    try:
        full_excel = export_full_excel(kpi, tables_dict, charts_dict_bytes, kpi_img)
        st.download_button("📥 Download All in ONE Excel", data=full_excel, file_name="dashboard_full_report.xlsx")
    except Exception as e:
        st.error(f"Failed to generate full Excel workbook: {e}")
        logger.exception("Failed to generate full Excel workbook")

if __name__ == '__main__':
    dashboard_module()
