# dashboard_module_full_color.py
import logging
from io import BytesIO
import logging
from io import BytesIO
from typing import Dict

import pandas as pd
import plotly.express as px
import streamlit as st
from fpdf import FPDF

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

# Currency symbol map
CURRENCY_MAP = {'$': 'USD', '€': 'EUR', '£': 'GBP', '₹': 'INR', '¥': 'JPY'}

# ---------------- Utilities ----------------
def preprocess_amounts(df):
    df['Currency'] = df['Balance Change'].astype(str).str.extract(r'([A-Z$€£₹¥]+)')[0].fillna('$')
    # Replace symbols with letters
    df['Currency'] = df['Currency'].map(CURRENCY_MAP).fillna('USD')
    # Remove non-numeric characters
    df['Balance Change'] = df['Balance Change'].astype(str).str.replace(r"[^\d\.-]", "", regex=True)
    df['Balance Change'] = pd.to_numeric(df['Balance Change'], errors='coerce').fillna(0.0)
    return df

def ensure_columns(df):
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
def compute_kpis(df):
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

# ---------------- Export Helpers ----------------
def to_excel(df_dict: Dict[str, pd.DataFrame], currency='$') -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet, df in df_dict.items():
            df_to_export = df.copy()
            for col in df_to_export.select_dtypes(include=['float', 'int']):
                df_to_export[col] = df_to_export[col].apply(lambda x: f"{currency}{x:,.2f}")
            df_to_export.to_excel(writer, index=False, sheet_name=sheet[:31])
            ws = writer.sheets[sheet[:31]]
            for i, col in enumerate(df_to_export.columns):
                width = max(df_to_export[col].astype(str).map(len).max(), len(col)) + 2
                ws.set_column(i, i, width)
    output.seek(0)
    return output.getvalue()

def export_pdf(df_dict: Dict[str, pd.DataFrame], kpis: Dict[str, float], currency='$') -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Accounting Dashboard Report', ln=True, align='C')
    pdf.ln(4)

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'Key Performance Indicators', ln=True)
    pdf.set_font('Arial', '', 10)
    for k, v in kpis.items():
        if isinstance(v, (int, float)) and v != float('inf'):
            pdf.cell(0, 6, f"{k}: {currency}{v:,.2f}", ln=True)
        else:
            pdf.cell(0, 6, f"{k}: {v}", ln=True)

    pdf.ln(4)
    for name, df in df_dict.items():
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, name, ln=True)
        pdf.set_font('Arial', '', 8)
        if df.empty:
            pdf.cell(0, 6, 'No data', ln=True)
            continue
        cols = list(df.columns)[:8]
        col_width = (pdf.w - 2*pdf.l_margin)/len(cols)
        th = 6
        for col in cols:
            pdf.cell(col_width, th, str(col)[:20], border=1)
        pdf.ln(th)
        for row in df[cols].values.tolist():
            for item in row:
                if isinstance(item, (int, float)):
                    pdf.cell(col_width, th, f"{currency}{item:,.2f}", border=1)
                else:
                    pdf.cell(col_width, th, str(item)[:20], border=1)
            pdf.ln(th)
    buf = BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf.read()

# ---------------- Journal & Ledger ----------------
def generate_journal_entries(df):
    journal = []
    df['main_lower'] = df['Main Category'].str.lower()
    for _, r in df.iterrows():
        amt = abs(float(r['Balance Change'] or 0))
        if amt == 0:
            continue
        date = r['Date']
        desc = r['Description']
        acct = f"{r['Main Category']} - {r['Subcategory']}"
        cat = r['main_lower']
        if cat in CATEGORY_DC_MAP:
            debit_acc, credit_acc = (CASH_ACCOUNT, acct) if CATEGORY_DC_MAP[cat][0]=='Credit' else (acct, CASH_ACCOUNT)
            journal.append({'Date': date, 'Description': desc,
                            'Debit Account': debit_acc, 'Debit': amt,
                            'Credit Account': credit_acc, 'Credit': amt})
        else:
            journal.append({'Date': date, 'Description': desc,
                            'Debit Account': acct, 'Debit': amt,
                            'Credit Account': CASH_ACCOUNT, 'Credit': amt})
    return pd.DataFrame(journal)

def build_ledger_from_journal(jdf):
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
    ledger['Running Balance'] = ledger.groupby('Account').apply(lambda g: (g['Debit']-g['Credit']).cumsum()).reset_index(level=0, drop=True)
    return ledger

# ---------------- Dashboard UI ----------------
def dashboard_module():
    st.set_page_config(page_title='Accounting Dashboard', layout='wide')
    st.title('📊 Accounting Dashboard (0% VAT — Real Double Entry)')

    if 'processed_df' not in st.session_state:
        st.error('No processed data found. Please upload data first.')
        return

    df = st.session_state['processed_df']
    df = preprocess_amounts(df)
    df = ensure_columns(df)

    # Sidebar
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

    # Tabs
    tabs = st.tabs(['Trial Balance','Income Statement','Balance Sheet','Cash Flow','P&L Monthly','Category Drilldown','Journal & Ledger'])

    # --- Trial Balance ---
    with tabs[0]:
        st.subheader('Trial Balance')
        tb = df_f.copy()
        def tb_dc(r):
            mc = str(r['Main Category']).lower()
            amt = abs(float(r['Balance Change'] or 0))
            if mc in ['expense','asset']:
                return pd.Series([amt,0.0])
            if mc in ['revenue','liability','equity']:
                return pd.Series([0.0,amt])
            return pd.Series([amt,0.0])
        tb[['Debit','Credit']] = tb.apply(tb_dc, axis=1)
        st.dataframe(tb[['Main Category','Subcategory','Debit','Credit']])
        st.plotly_chart(px.bar(tb, x='Main Category', y='Balance Change', color='Main Category', 
                               color_discrete_sequence=CHART_COLORS), key='trial_balance_bar')
        ttl_d, ttl_c = tb['Debit'].sum(), tb['Credit'].sum()
        if abs(ttl_d-ttl_c)<0.01:
            st.success(f"Balanced ✓ Debit={ttl_d:,.2f} Credit={ttl_c:,.2f}")
        else:
            st.error(f"NOT Balanced ✗ Debit={ttl_d:,.2f} Credit={ttl_c:,.2f}")

    # --- Income Statement ---
    with tabs[1]:
        st.subheader('Income Statement')
        inc = df_f[df_f['Main Category'].str.lower().isin(['revenue','expense'])]
        if not inc.empty:
            g = inc.groupby('Main Category')['Balance Change'].sum().reset_index()
            st.dataframe(g)
            st.plotly_chart(px.bar(g, x='Main Category', y='Balance Change', color='Main Category',
                                   color_discrete_sequence=CHART_COLORS), key='income_statement_bar')

    # --- Balance Sheet ---
    with tabs[2]:
        st.subheader('Balance Sheet')
        bs = df_f[df_f['Main Category'].str.lower().isin(['asset','liability','equity'])]
        if not bs.empty:
            bs_sum = bs.groupby('Main Category')['Balance Change'].sum().reset_index()
            st.dataframe(bs_sum)
            st.plotly_chart(px.bar(bs_sum, x='Main Category', y='Balance Change', color='Main Category',
                                   color_discrete_sequence=CHART_COLORS), key='balance_sheet_bar')
            if abs(kpi['Total Assets']-(kpi['Total Liabilities']+kpi['Total Equity']))<0.01:
                st.success('Balanced ✓ Assets = Liabilities + Equity')
            else:
                st.error('NOT Balanced ✗ Asset equation incorrect')

    # --- Cash Flow ---
    with tabs[3]:
        st.subheader('Cash Flow')
        cf = df_f.copy()
        cf['Month'] = cf['Date'].dt.to_period('M').astype(str)
        cf['Type'] = cf['Main Category'].apply(lambda mc:'Cash In' if mc.lower()=='revenue' else ('Cash Out' if mc.lower()=='expense' else ('Financing' if mc.lower() in ['equity','liability'] else 'Other')))
        g = cf.groupby(['Month','Type'])['Balance Change'].sum().reset_index()
        st.dataframe(g)
        pivot = g.pivot(index='Month', columns='Type', values='Balance Change').fillna(0).reset_index()
        st.plotly_chart(px.bar(pivot, x='Month', y=pivot.columns[1:].tolist(), barmode='group',
                               color_discrete_sequence=CHART_COLORS), key='cash_flow_bar')

    # --- P&L Monthly ---
    with tabs[4]:
        st.subheader('Profit & Loss Monthly')
        pl = df_f[df_f['Main Category'].str.lower().isin(['revenue','expense'])].copy()
        if not pl.empty:
            pl['Month'] = pl['Date'].dt.to_period('M')
            g = pl.groupby([pl['Month'].astype(str),'Main Category'])['Balance Change'].sum().unstack(fill_value=0)
            for col in ['Revenue','Expense']:
                if col not in g.columns:
                    g[col]=0.0
            g.index=pd.PeriodIndex(g.index,freq='M').to_timestamp()
            g['Net Profit']=g['Revenue']-g['Expense']
            st.dataframe(g.reset_index())
            plot_df=g.reset_index()
            plot_df['Month']=plot_df['Month'].dt.to_period('M').astype(str)
            st.plotly_chart(px.bar(plot_df,x='Month',y=['Revenue','Expense'], barmode='group',
                                   color_discrete_sequence=CHART_COLORS), key='pl_monthly_bar')
            st.plotly_chart(px.line(plot_df,x='Month',y='Net Profit', markers=True, color_discrete_sequence=CHART_COLORS),
                            key='pl_monthly_line')

    # --- Category Drilldown ---
    with tabs[5]:
        st.subheader('Category Drilldown')
        mc=df_f.groupby('Main Category')['Balance Change'].sum().reset_index()
        if not mc.empty:
            st.plotly_chart(px.bar(mc, x='Main Category', y='Balance Change', color='Main Category',
                                   color_discrete_sequence=CHART_COLORS), key='category_drilldown_bar')
            choose=st.selectbox('Select Category',['']+mc['Main Category'].tolist())
            if choose:
                sub=df_f[df_f['Main Category']==choose].groupby('Subcategory')['Balance Change'].sum().reset_index()
                st.plotly_chart(px.bar(sub,x='Subcategory',y='Balance Change', color='Subcategory',
                                       color_discrete_sequence=CHART_COLORS), key=f'subcategory_drilldown_{choose}')
                st.dataframe(sub)

    # --- Journal & Ledger ---
    with tabs[6]:
        st.subheader('Journal Entries')
        jdf=generate_journal_entries(df_f)
        st.dataframe(jdf)
        st.subheader('Ledger')
        ledger=build_ledger_from_journal(jdf)
        st.dataframe(ledger)
        if not ledger.empty:
            st.subheader('Ledger Summary')
            s=ledger.groupby('Account')[['Debit','Credit']].sum()
            s['Closing Balance']=s['Debit']-s['Credit']
            st.dataframe(s.reset_index())
            st.subheader('Running Balance Chart')
            acct=st.selectbox('Select Account', s.index.tolist())
            if acct:
                adf=ledger[ledger['Account']==acct]
                st.plotly_chart(px.line(adf,x='Date',y='Running Balance',markers=True,
                                        color_discrete_sequence=CHART_COLORS), key=f'running_balance_{acct}')

    # --- Downloads ---
    st.subheader('Download Reports')
    df_dict={'Filtered Data': df_f.reset_index(drop=True)}
    st.download_button('Download Excel', data=to_excel(df_dict, selected_currency), file_name='dashboard_report.xlsx')
    st.download_button('Download PDF', data=export_pdf(df_dict,kpi,selected_currency), file_name='dashboard_report.pdf')


if __name__=='__main__':
    dashboard_module()