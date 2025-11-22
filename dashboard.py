import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
from fpdf import FPDF
from fileupload import get_processed_df  # your file processing function

# ---------------- Load Processed Data ---------------- #
def load_data():
    if 'processed_df' not in st.session_state:
        st.error("No processed data found. Please upload your transactions first.")
        return pd.DataFrame()
    df = st.session_state['processed_df'].copy()
    if 'Date' not in df.columns:
        df['Date'] = pd.Timestamp.today()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').fillna(pd.Timestamp.today())
    essential_cols = ['Main Category', 'Subcategory', 'Balance Change', 'Description', 'Auto-Matched', 'Date']
    for col in essential_cols:
        if col not in df.columns:
            if col == 'Balance Change': df[col] = 0
            elif col == 'Auto-Matched': df[col] = True
            elif col == 'Date': df[col] = pd.Timestamp.today()
            else: df[col] = 'Unknown'
    df = df[essential_cols]
    return df

# ---------------- KPIs ---------------- #
def compute_kpis(df):
    revenue = df[df['Main Category']=='Revenue']['Balance Change'].sum()
    expense = df[df['Main Category']=='Expense']['Balance Change'].sum()
    net_profit = revenue - expense
    total_assets = df[df['Main Category']=='Asset']['Balance Change'].sum()
    total_liabilities = df[df['Main Category']=='Liability']['Balance Change'].sum()
    total_equity = df[df['Main Category']=='Equity']['Balance Change'].sum()
    current_ratio = total_assets / total_liabilities if total_liabilities else 0
    return {'Revenue': revenue, 'Expense': expense, 'Net Profit': net_profit,
            'Total Assets': total_assets, 'Total Liabilities': total_liabilities,
            'Total Equity': total_equity, 'Current Ratio': current_ratio}

# ---------------- Export Functions ---------------- #
def to_excel(df_dict):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in df_dict.items():
            df.to_excel(writer, index=False, sheet_name=sheet_name)
            for i, col in enumerate(df.columns):
                col_width = max(df[col].astype(str).map(len).max(), len(col)) + 2
                writer.sheets[sheet_name].set_column(i, i, col_width)
    output.seek(0)
    return output.getvalue()

def export_pdf(df_dict, kpis):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, 'Professional Accounting Dashboard Report', ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, 'Key Performance Indicators', ln=True)
    pdf.set_font("Arial", '', 10)
    for k, v in kpis.items():
        pdf.cell(0, 8, f'{k}: ${v:,.2f}' if isinstance(v,float) else f'{k}: {v}', ln=True)
    pdf.ln(5)
    for name, df in df_dict.items():
        if df.empty: continue
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0,10,name,ln=True)
        pdf.set_font("Arial",'',8)
        col_width = pdf.w / (len(df.columns)+1)
        th = 6
        for col in df.columns:
            pdf.cell(col_width, th, str(col), border=1, align='C', fill=True)
        pdf.ln(th)
        for row in df.values.tolist():
            for item in row:
                text = str(item).replace("–","-").replace("—","-")
                pdf.cell(col_width, th, text, border=1)
            pdf.ln(th)
        pdf.ln(5)
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output.read()

# ---------------- Accounting Helper ---------------- #
def assign_debit_credit(row):
    if row['Main Category'] in ['Expense','Asset']:
        debit = abs(row['Balance Change'])
        credit = 0
    elif row['Main Category'] in ['Revenue','Liability','Equity']:
        debit = 0
        credit = abs(row['Balance Change'])
    else:
        debit = max(row['Balance Change'],0)
        credit = -min(row['Balance Change'],0)
    return pd.Series([debit, credit])

# ---------------- Dashboard Module ---------------- #
def dashboard_module():
    st.title("📊 Professional Accounting Dashboard (Production Ready)")

    df = load_data()
    if df.empty: return

    # Sidebar Filters
    st.sidebar.header("Filters")
    start_date = st.sidebar.date_input("Start Date", df['Date'].min())
    end_date = st.sidebar.date_input("End Date", df['Date'].max())
    main_categories = st.sidebar.multiselect("Main Category", df['Main Category'].unique())
    sub_categories = st.sidebar.multiselect("Subcategory", df['Subcategory'].unique())
    keyword = st.sidebar.text_input("Search Description")
    df_filtered = df[(df['Date']>=pd.to_datetime(start_date)) & (df['Date']<=pd.to_datetime(end_date))]
    if main_categories: df_filtered=df_filtered[df_filtered['Main Category'].isin(main_categories)]
    if sub_categories: df_filtered=df_filtered[df_filtered['Subcategory'].isin(sub_categories)]
    if keyword: df_filtered=df_filtered[df_filtered['Description'].str.contains(keyword,case=False,na=False)]
    if df_filtered.empty: st.warning("No transactions found."); return

    # KPIs
    kpis = compute_kpis(df_filtered)
    st.subheader("Key Performance Indicators")
    cols = st.columns(3)
    cols[0].metric("Revenue", f"${kpis['Revenue']:,.2f}")
    cols[1].metric("Expense", f"${kpis['Expense']:,.2f}")
    cols[2].metric("Net Profit", f"${kpis['Net Profit']:,.2f}")
    cols = st.columns(3)
    cols[0].metric("Total Assets", f"${kpis['Total Assets']:,.2f}")
    cols[1].metric("Total Liabilities", f"${kpis['Total Liabilities']:,.2f}")
    cols[2].metric("Total Equity", f"${kpis['Total Equity']:,.2f}")
    st.metric("Current Ratio", f"{kpis['Current Ratio']:.2f}")

    # Tabs
    tabs = st.tabs(["Trial Balance","Income Statement","Balance Sheet","Cash Flow","P&L","Category Drilldown","Journal & Ledger"])

    # ---------- Trial Balance ----------
    with tabs[0]:
        st.subheader("Trial Balance")
        trial_cols=['Main Category','Subcategory','Balance Change']
        trial=df_filtered[trial_cols].copy()
        trial[['Debit','Credit']] = trial.apply(assign_debit_credit,axis=1)
        trial['Status'] = trial.apply(lambda r: 'Balanced' if abs(r['Debit']-r['Credit'])<0.01 else 'Check',axis=1)
        st.dataframe(trial[['Main Category','Subcategory','Debit','Credit','Balance Change','Status']])

        # Main Category Chart
        trial_summary = trial.groupby('Main Category')[['Debit','Credit']].sum().reset_index()
        fig=px.bar(trial_summary,x='Main Category',y=['Debit','Credit'],barmode='group',
                   title='Trial Balance by Main Category',text_auto='.2f',color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig,use_container_width=True)

        total_debit=trial['Debit'].sum(); total_credit=trial['Credit'].sum()
        if abs(total_debit-total_credit)<0.01: st.success(f"✅ Total Trial Balance is balanced (Debit={total_debit:,.2f}, Credit={total_credit:,.2f})")
        else: st.error(f"⚠️ Total Trial Balance NOT balanced! Debit={total_debit:,.2f}, Credit={total_credit:,.2f}")

    # ---------- Income Statement ----------
    with tabs[1]:
        st.subheader("Income Statement")
        income=df_filtered[df_filtered['Main Category'].isin(['Revenue','Expense'])]
        if not income.empty:
            # Main Category Table & Chart
            main_summary = income.groupby('Main Category')['Balance Change'].sum().reset_index()
            st.markdown("**By Main Category**")
            st.dataframe(main_summary)
            fig_main=px.bar(main_summary,x='Main Category',y='Balance Change',color='Main Category',
                            title='Income Statement - Main Category',text='Balance Change',color_discrete_sequence=px.colors.qualitative.Set3)
            fig_main.update_traces(texttemplate='%{text:,.2f}',textposition='outside')
            st.plotly_chart(fig_main,use_container_width=True)

            # Subcategory Table & Chart
            sub_summary = income.groupby('Subcategory')['Balance Change'].sum().reset_index()
            st.markdown("**By Subcategory**")
            st.dataframe(sub_summary)
            fig_sub=px.bar(sub_summary,x='Subcategory',y='Balance Change',color='Subcategory',
                           title='Income Statement - Subcategory',text='Balance Change',color_discrete_sequence=px.colors.qualitative.Vivid)
            fig_sub.update_traces(texttemplate='%{text:,.2f}',textposition='outside')
            st.plotly_chart(fig_sub,use_container_width=True)

    # ---------- Balance Sheet ----------
    with tabs[2]:
        st.subheader("Balance Sheet")
        bs=df_filtered[df_filtered['Main Category'].isin(['Asset','Liability','Equity'])]
        # Ensure table always has main categories
        categories = ['Asset','Liability','Equity']
        bs_summary = bs.groupby('Main Category')['Balance Change'].sum().reindex(categories, fill_value=0).reset_index()
        st.dataframe(bs_summary)
        fig_bs=px.bar(bs_summary,x='Main Category',y='Balance Change',color='Main Category',
                      title='Balance Sheet Overview',color_discrete_sequence=px.colors.qualitative.Vivid)
        st.plotly_chart(fig_bs,use_container_width=True)
        if abs(kpis['Total Assets']-(kpis['Total Liabilities']+kpis['Total Equity']))<0.01:
            st.success("✅ Balance Sheet is balanced (Assets = Liabilities + Equity)")
        else:
            st.error("⚠️ Balance Sheet NOT balanced!")

    # ---------- Cash Flow ----------
    with tabs[3]:
        st.subheader("Cash Flow")
        cash_df = df_filtered[df_filtered['Main Category'].isin(['Revenue','Expense','Liability','Equity'])].copy()
        if not cash_df.empty:
            # Classify Cash In / Out
            cash_df['Type'] = cash_df['Main Category'].apply(lambda x:'AR (Cash In)' if x=='Revenue' else 'AP (Cash Out)')
            cash_df['Month'] = cash_df['Date'].dt.to_period('M').astype(str)
            cash_flow = cash_df.groupby(['Month','Type'])['Balance Change'].sum().reset_index()
            st.dataframe(cash_flow)
            fig_cash=px.bar(cash_flow,x='Month',y='Balance Change',color='Type',barmode='group',
                            title='Monthly Cash Flow (AR vs AP)',text='Balance Change',
                            color_discrete_sequence=px.colors.qualitative.Set2)
            fig_cash.update_traces(texttemplate='%{text:,.2f}',textposition='outside')
            st.plotly_chart(fig_cash,use_container_width=True)

    # ---------- P&L ----------
    with tabs[4]:
        st.subheader("Profit & Loss Statement")
        pnl=df_filtered[df_filtered['Main Category'].isin(['Revenue','Expense'])]
        if not pnl.empty:
            pnl['Month']=pnl['Date'].dt.to_period('M').astype(str)
            pnl_summary=pnl.groupby(['Month','Main Category'])['Balance Change'].sum().unstack().fillna(0)
            pnl_summary['Net Profit']=pnl_summary.get('Revenue',0)-pnl_summary.get('Expense',0)
            pnl_summary=pnl_summary.reset_index()
            st.dataframe(pnl_summary)
            colors = ['green' if x>=0 else 'red' for x in pnl_summary['Net Profit']]
            fig_pnl = px.bar(pnl_summary,x='Month',y=['Revenue','Expense'],barmode='group',text_auto='.2f',
                             title='Monthly Revenue vs Expense',color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_pnl,use_container_width=True)
            fig_net=px.line(pnl_summary,x='Month',y='Net Profit',markers=True,
                            title='Monthly Net Profit Trend',color_discrete_sequence=px.colors.sequential.Viridis)
            st.plotly_chart(fig_net,use_container_width=True)

    # ---------- Category Drilldown ----------
    with tabs[5]:
        st.subheader("Category Drilldown")
        main_cat_summary=df_filtered.groupby('Main Category')['Balance Change'].sum().reset_index()
        fig=px.bar(main_cat_summary,x='Main Category',y='Balance Change',color='Main Category',
                    text='Balance Change',title="Main Category Overview",color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_traces(texttemplate='%{text:,.2f}', textposition='outside')
        st.plotly_chart(fig,use_container_width=True)

        selected_main_cat=st.selectbox("Select Main Category for Drilldown",['']+main_cat_summary['Main Category'].tolist())
        if selected_main_cat:
            subcat_df=df_filtered[df_filtered['Main Category']==selected_main_cat]
            subcat_summary=subcat_df.groupby('Subcategory')['Balance Change'].sum().reset_index()
            fig2=px.bar(subcat_summary,x='Subcategory',y='Balance Change',color='Subcategory',
                        text='Balance Change',title=f"Subcategories under {selected_main_cat}",color_discrete_sequence=px.colors.qualitative.Vivid)
            fig2.update_traces(texttemplate='%{text:,.2f}',textposition='outside')
            st.plotly_chart(fig2,use_container_width=True)
            st.dataframe(subcat_summary)

    # ---------- Journal & Ledger ----------
    with tabs[6]:
        st.subheader("📘 Journal & Ledger")
        jdf=df_filtered.copy()
        if 'Description' not in jdf.columns: jdf['Description']="No Description"
        if 'Account' not in jdf.columns: jdf['Account']=jdf['Main Category'].astype(str)+' - '+jdf['Subcategory'].astype(str)
        journal_entries=[]
        for _,row in jdf.iterrows():
            amount=float(row['Balance Change'])
            if row['Main Category'] in ['Expense','Asset']:
                journal_entries.append({"Date":row['Date'],"Description":row['Description'],
                                        "Debit Account":row['Account'],"Debit":abs(amount),
                                        "Credit Account":"System Auto Offset","Credit":abs(amount)})
            elif row['Main Category'] in ['Revenue','Liability','Equity']:
                journal_entries.append({"Date":row['Date'],"Description":row['Description'],
                                        "Debit Account":"System Auto Offset","Debit":abs(amount),
                                        "Credit Account":row['Account'],"Credit":abs(amount)})
        journal_df=pd.DataFrame(journal_entries)
        st.subheader("📄 Journal Entries")
        st.dataframe(journal_df)
        # Ledger
        ledger_rows=[]
        for _,row in journal_df.iterrows():
            ledger_rows.append({"Account":row["Debit Account"],"Date":row["Date"],
                                "Description":row["Description"],"Debit":row["Debit"],"Credit":0})
            ledger_rows.append({"Account":row["Credit Account"],"Date":row["Date"],
                                "Description":row["Description"],"Debit":0,"Credit":row["Credit"]})
        ledger_df=pd.DataFrame(ledger_rows)
        ledger_df.sort_values(["Account","Date"],inplace=True)
        ledger_df["Running Balance"]=ledger_df.groupby("Account").apply(lambda x:(x["Debit"]-x["Credit"]).cumsum()).reset_index(level=0,drop=True)
        st.subheader("📚 Ledger")
        st.dataframe(ledger_df)
        summary=ledger_df.groupby("Account").agg({"Debit":"sum","Credit":"sum"}).reset_index()
        summary["Closing Balance"]=summary["Debit"]-summary["Credit"]
        st.subheader("📘 Ledger Summary")
        st.dataframe(summary)
        st.subheader("📈 Ledger Balance Charts")
        selected_acct=st.selectbox("Select Account",summary["Account"].unique())
        acct_df=ledger_df[ledger_df["Account"]==selected_acct]
        fig=px.line(acct_df,x="Date",y="Running Balance",title=f"Running Balance - {selected_acct}",markers=True)
        st.plotly_chart(fig,use_container_width=True)

    # ---------------- Download Reports ---------------- #
    st.markdown("---")
    st.subheader("📥 Download Reports")
    df_dict={'Raw Data':df_filtered,'Main Category Summary':main_cat_summary}
    if 'selected_main_cat' in locals() and selected_main_cat: df_dict[f'{selected_main_cat} Subcategories']=subcat_summary
    st.download_button("⬇️ Download Excel Report",data=to_excel(df_dict),file_name="dashboard_report.xlsx")
    st.download_button("⬇️ Download PDF Report",data=export_pdf(df_dict,kpis),file_name="dashboard_report.pdf")

if __name__=="__main__":
    dashboard_module()
