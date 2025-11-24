import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from fpdf import FPDF
from datetime import datetime
import kaleido

# ---------------- Load Processed Data ---------------- #
def load_data():
    if 'processed_df' not in st.session_state:
        st.error("No processed data found. Please upload your transactions first.")
        return pd.DataFrame()
    df = st.session_state['processed_df'].copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').fillna(pd.Timestamp.today())
    else:
        df['Date'] = pd.Timestamp.today()
    for col in ['Main Category', 'Subcategory', 'Balance Change']:
        if col not in df.columns:
            df[col] = 0 if col == 'Balance Change' else 'Unknown'
    df['Balance Change'] = pd.to_numeric(df['Balance Change'], errors='coerce').fillna(0.0)
    return df

# ---------------- PDF Export with Charts & Cover ---------------- #
class PDFReport(FPDF):
    def header(self):
        # corporate style header
        self.set_font("Arial", "B", 12)
        self.set_text_color(0, 0, 128)
        self.cell(0, 10, "Budget & Forecast Report", ln=True, align='C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 10)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align='C')

def generate_pdf(actual_summary, cumulative, forecast_display, fig_actual, fig_cum, fig_forecast):
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # ---- Cover Page ----
    pdf.add_page()
    pdf.set_font("Arial", 'B', 24)
    pdf.set_text_color(0, 0, 128)
    pdf.cell(0, 20, "Corporate Budget & Forecast", ln=True, align='C')
    pdf.set_font("Arial", '', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"Total Actual: ${actual_summary['Balance Change'].sum():,.2f}", ln=True, align='L')
    pdf.cell(0, 10, f"Total Budget: ${actual_summary['Budgeted'].sum():,.2f}", ln=True, align='L')
    pdf.cell(0, 10, f"Total Variance: ${actual_summary['Balance Change'].sum() - actual_summary['Budgeted'].sum():,.2f}", ln=True, align='L')

    # ---- Actual vs Budget ----
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0, 0, 128)
    pdf.cell(0, 10, "1. Actual vs Budget", ln=True)
    pdf.ln(2)
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(200, 200, 200)
    pdf.cell(60, 8, "Category", 1, 0, 'C', 1)
    pdf.cell(40, 8, "Actual ($)", 1, 0, 'C', 1)
    pdf.cell(40, 8, "Budget ($)", 1, 0, 'C', 1)
    pdf.cell(40, 8, "Variance ($)", 1, 0, 'C', 1)
    pdf.cell(30, 8, "Variance %", 1, 1, 'C', 1)
    pdf.set_font("Arial", '', 11)
    for idx, row in actual_summary.iterrows():
        pdf.set_text_color(0, 128, 0) if row['Variance'] >= 0 else pdf.set_text_color(255, 0, 0)
        pdf.cell(60, 6, f"{row['Main Category']} / {row['Subcategory']}", 1)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(40, 6, f"{row['Balance Change']:.2f}", 1, 0, 'R')
        pdf.cell(40, 6, f"{row['Budgeted']:.2f}", 1, 0, 'R')
        pdf.set_text_color(0, 128, 0) if row['Variance'] >= 0 else pdf.set_text_color(255, 0, 0)
        pdf.cell(40, 6, f"{row['Variance']:.2f}", 1, 0, 'R')
        pdf.cell(30, 6, f"{row['Variance %']:.1f}%", 1, 1, 'R')

    # Embed Actual vs Budget chart
    pdf.ln(5)
    buf1 = BytesIO()
    fig_actual.write_image(buf1, format='png')
    buf1.seek(0)
    pdf.image(buf1, w=180)
    
    # ---- Cumulative ----
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0, 0, 128)
    pdf.cell(0, 10, "2. Cumulative Spending", ln=True)
    pdf.ln(2)
    pdf.set_font("Arial", '', 11)
    for idx, row in cumulative.iterrows():
        pdf.cell(0, 6, f"{row['Month']} - {row['Main Category']} / {row['Subcategory']}: ${row['Cumulative']:.2f}", ln=True)
    buf2 = BytesIO()
    fig_cum.write_image(buf2, format='png')
    buf2.seek(0)
    pdf.image(buf2, w=180)

    # ---- Forecast ----
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0, 0, 128)
    pdf.cell(0, 10, "3. Forecast", ln=True)
    pdf.ln(2)
    pdf.set_font("Arial", '', 11)
    for idx, row in forecast_display.iterrows():
        pdf.cell(0, 6, f"{row['Month']} - {row['Main Category']} / {row['Subcategory']}: Forecast=${row['Forecast']:.2f}", ln=True)
    buf3 = BytesIO()
    fig_forecast.write_image(buf3, format='png')
    buf3.seek(0)
    pdf.image(buf3, w=180)

    output = BytesIO()
    pdf.output(output)
    output.seek(0)
    return output

# ---------------- Budget Module ---------------- #
def budget_module():
    st.title("📊 QuickBooks-Style Budgeting & Forecasting (Professional PDF)")

    df = load_data()
    if df.empty:
        return

    # Filters
    st.sidebar.header("Filters")
    start_date = st.sidebar.date_input("Start Date", df['Date'].min())
    end_date = st.sidebar.date_input("End Date", df['Date'].max())
    main_categories = st.sidebar.multiselect("Main Category", df['Main Category'].unique())
    sub_categories = st.sidebar.multiselect("Subcategory", df['Subcategory'].unique())

    df_filtered = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
    if main_categories:
        df_filtered = df_filtered[df_filtered['Main Category'].isin(main_categories)]
    if sub_categories:
        df_filtered = df_filtered[df_filtered['Subcategory'].isin(sub_categories)]
    if df_filtered.empty:
        st.info("No transactions found for selected filters.")
        return

    df_filtered['Balance Change'] = df_filtered.apply(
        lambda x: abs(x['Balance Change']) if x['Main Category'].lower() == 'expense' else x['Balance Change'], axis=1
    )

    # Budget input
    st.subheader(" Set / Edit Budgets per Category")
    budgeted = {}
    actual_summary = df_filtered.groupby(['Main Category', 'Subcategory'])['Balance Change'].sum().reset_index()
    for idx, row in actual_summary.iterrows():
        main_cat, sub_cat, actual = row['Main Category'], row['Subcategory'], row['Balance Change']
        budgeted[(main_cat, sub_cat)] = st.number_input(
            f"Budget for {main_cat} / {sub_cat}",
            value=float(actual),
            step=0.01
        )
    actual_summary['Budgeted'] = actual_summary.apply(
        lambda x: budgeted.get((x['Main Category'], x['Subcategory']), 0), axis=1
    )
    def compute_variance(row):
        return (row['Budgeted'] - row['Balance Change']) if row['Main Category'].lower() == 'expense' else (row['Balance Change'] - row['Budgeted'])
    actual_summary['Variance'] = actual_summary.apply(compute_variance, axis=1)
    actual_summary['Variance %'] = actual_summary.apply(lambda x: (x['Variance']/x['Budgeted']*100) if x['Budgeted']!=0 else 0, axis=1)

    st.subheader(" Actual vs Budget Summary")
    st.dataframe(actual_summary.style.applymap(lambda v: 'color: green' if v>=0 else 'color: red', subset=['Variance']))

    # Charts
    df_filtered['Month'] = df_filtered['Date'].dt.to_period('M').astype(str)
    cumulative = df_filtered.groupby(['Month','Main Category','Subcategory'])['Balance Change'].sum().reset_index()
    cumulative['Cumulative'] = cumulative.groupby(['Main Category','Subcategory'])['Balance Change'].cumsum()
    fig_cum = px.line(cumulative, x='Month', y='Cumulative', color='Subcategory', line_dash='Main Category', markers=True, title='Cumulative Spending')
    st.plotly_chart(fig_cum, use_container_width=True)
    fig_actual = px.bar(actual_summary, x='Subcategory', y=['Balance Change','Budgeted'], color='Main Category', barmode='group', text_auto=True, title='Actual vs Budget')
    st.plotly_chart(fig_actual, use_container_width=True)

    # Forecast
    st.subheader("🔮 Forecast (Weighted Moving Average)")
    forecast_months = st.number_input("Forecast Months", min_value=1, max_value=12, value=3)
    period = st.number_input("Moving Average Period", min_value=1, max_value=12, value=3)
    forecast_display = pd.DataFrame()
    try:
        forecast_df = df_filtered.groupby(['Month','Main Category','Subcategory'])['Balance Change'].sum().reset_index()
        forecast_list = []
        for idx, cat_df in forecast_df.groupby(['Main Category','Subcategory']):
            cat_df = cat_df.sort_values('Month')
            cat_df['Forecast'] = cat_df['Balance Change'].rolling(period, min_periods=1).apply(lambda x: np.average(x, weights=np.arange(1,len(x)+1)), raw=True)
            last_forecast = cat_df['Forecast'].iloc[-1] if not cat_df.empty else 0
            main_cat, sub_cat = idx
            growth_rate = cat_df['Balance Change'].pct_change().fillna(0).mean()
            for i in range(1, forecast_months+1):
                projected = last_forecast * (1 + growth_rate)
                forecast_list.append({'Month':f'Next Month +{i}','Main Category':main_cat,'Subcategory':sub_cat,'Forecast':projected})
                last_forecast = projected
        forecast_display = pd.DataFrame(forecast_list)
        fig_forecast = px.line(forecast_display, x='Month', y='Forecast', color='Subcategory', line_dash='Main Category', markers=True, title='Forecast')
        st.plotly_chart(fig_forecast, use_container_width=True)
    except Exception as e:
        st.warning(f"Forecasting failed: {e}")
        fig_forecast = px.line(title='Forecast (No Data)')

    # Download PDF
    st.download_button(
        "⬇️ Download Professional PDF Report",
        data=generate_pdf(actual_summary, cumulative, forecast_display, fig_actual, fig_cum, fig_forecast),
        file_name="budget_forecast_professional.pdf"
    )

# ---------------- Main ---------------- #
if __name__ == "__main__":
    budget_module()
