import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from fpdf import FPDF
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

# --------------------- Utility: DataFrame to Table Image --------------------- #
def df_to_image(df, title=None, col_width=120, row_height=30,
                header_color=(0, 0, 128),
                row_colors=[(255, 255, 255), (230, 230, 250)],
                max_rows_per_page=25):
    """Convert a pandas DataFrame to colorful images split by pages if needed"""
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()

    images = []
    rows, cols = df.shape

    for start_row in range(0, rows, max_rows_per_page):
        end_row = min(start_row + max_rows_per_page, rows)
        page_rows = end_row - start_row

        width = col_width * cols
        height = row_height * (page_rows + 1) + 50

        img = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Title
        y_offset = 0
        if title:
            draw.text((10, 5), title, fill=(0, 0, 128), font=font)
            y_offset = 50

        # Header
        for j, col in enumerate(df.columns):
            draw.rectangle([j * col_width, y_offset, (j + 1) * col_width, y_offset + row_height],
                           fill=header_color)
            draw.text((j * col_width + 5, y_offset + 5), str(col), fill=(255, 255, 255), font=small_font)

        y = y_offset + row_height

        # Rows
        for i in range(start_row, end_row):
            row = df.iloc[i]
            fill = row_colors[i % 2]
            draw.rectangle([0, y, width, y + row_height], fill=fill)
            for j, col in enumerate(df.columns):
                value = row[col]
                if isinstance(value, (int, float, np.integer, np.floating)):
                    text = f"{value:,.2f}"
                else:
                    text = str(value)
                draw.text((j * col_width + 5, y + 5), text, fill=(0, 0, 0), font=small_font)
            y += row_height

        images.append(img)

    # If DataFrame empty, return a single blank image
    if not images:
        img = Image.new("RGB", (col_width * max(1, len(df.columns)), 100), color=(255, 255, 255))
        images.append(img)

    return images


# --------------------- PDF Class --------------------- #
class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.set_text_color(0, 0, 128)
        self.cell(0, 10, "Budget & Forecast Report", ln=True, align='C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 10)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align='C')


# --------------------- PDF Generation --------------------- #
def generate_pdf(df_summary, df_cumulative, df_forecast, fig_actual, fig_cum, fig_forecast):
    def safe_fig_to_buffer(fig, width=900, height=500):
        buf = BytesIO()
        try:
            fig.write_image(buf, format="png", engine="kaleido", width=width, height=height)
        except Exception as e:
            fallback = Image.new("RGB", (width, height), color=(255, 255, 255))
            d = ImageDraw.Draw(fallback)
            d.text((20, 20), f"Chart unavailable:\n{e}", fill=(0, 0, 0))
            fallback.save(buf, format="PNG")
        buf.seek(0)
        return buf

    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Summary table images
    table_images = df_to_image(df_summary, title="Actual vs Budget Summary", max_rows_per_page=25)
    for img in table_images:
        pdf.add_page()
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        pdf.image(buf, w=180)

    # Charts
    pdf.add_page()
    pdf.image(safe_fig_to_buffer(fig_actual), w=180)

    pdf.add_page()
    pdf.image(safe_fig_to_buffer(fig_cum), w=180)

    pdf.add_page()
    pdf.image(safe_fig_to_buffer(fig_forecast), w=180)

    output = BytesIO()
    pdf.output(output)
    output.seek(0)
    return output


# --------------------- Load Processed Data --------------------- #
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


# --------------------- Budget Module --------------------- #
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
        lambda x: abs(x['Balance Change']) if str(x.get('Main Category', '')).lower() == 'expense' else x['Balance Change'],
        axis=1
    )

    # Budget input
    st.subheader("Set / Edit Budgets per Category")
    budgeted = {}
    actual_summary = df_filtered.groupby(['Main Category', 'Subcategory'])['Balance Change'].sum().reset_index()

    for idx, row in actual_summary.iterrows():
        main_cat = row['Main Category']
        sub_cat = row['Subcategory']
        actual = row['Balance Change']
        budgeted[(main_cat, sub_cat)] = st.number_input(
            f"Budget for {main_cat} / {sub_cat}",
            value=float(actual),
            step=0.01,
            key=f"budget_{idx}"
        )

    actual_summary['Budgeted'] = actual_summary.apply(
        lambda x: budgeted.get((x['Main Category'], x['Subcategory']), 0), axis=1
    )

    def compute_variance(row):
        if str(row['Main Category']).lower() == 'expense':
            return row['Budgeted'] - row['Balance Change']
        return row['Balance Change'] - row['Budgeted']

    actual_summary['Variance'] = actual_summary.apply(compute_variance, axis=1)
    actual_summary['Variance %'] = actual_summary.apply(
        lambda x: (x['Variance'] / x['Budgeted'] * 100) if x['Budgeted'] != 0 else 0, axis=1
    )

    st.subheader("Actual vs Budget Summary")
    st.dataframe(actual_summary.style.format({
        'Balance Change': '${:,.2f}',
        'Budgeted': '${:,.2f}',
        'Variance': '${:,.2f}',
        'Variance %': '{:.1f}%'
    }))

    # Charts
    df_filtered['Month'] = df_filtered['Date'].dt.to_period('M').astype(str)
    cumulative = df_filtered.groupby(['Month', 'Main Category', 'Subcategory'])['Balance Change'].sum().reset_index()
    cumulative['Cumulative'] = cumulative.groupby(['Main Category', 'Subcategory'])['Balance Change'].cumsum()

    fig_cum = px.line(
    cumulative,
    x='Month',
    y='Cumulative',
    color='Subcategory',
    line_dash='Main Category',
    markers=True,
    title='Cumulative Spending',
    color_discrete_sequence=px.colors.qualitative.Set2     # 🌈 colorful palette
)

    st.plotly_chart(fig_cum, use_container_width=True)

    fig_actual = px.bar(
    actual_summary,
    x='Subcategory',
    y=['Balance Change', 'Budgeted'],
    color='Main Category',
    barmode='group',
    text_auto=True,
    title='Actual vs Budget',
    color_discrete_map={
        'Income': '#1f77b4',       # blue
        'Expense': '#ff7f0e',      # orange
        'Other': '#2ca02c'
    }
)

    st.plotly_chart(fig_actual, use_container_width=True)

    # Forecast
    st.subheader("🔮 Forecast (Weighted Moving Average)")
    forecast_months = st.number_input("Forecast Months", min_value=1, max_value=12, value=3)
    period = st.number_input("Moving Average Period", min_value=1, max_value=12, value=3)
    forecast_display = pd.DataFrame()
    try:
        forecast_df = df_filtered.groupby(['Month', 'Main Category', 'Subcategory'])['Balance Change'].sum().reset_index()
        forecast_list = []

        for idx, cat_df in forecast_df.groupby(['Main Category', 'Subcategory']):
            cat_df = cat_df.sort_values('Month').reset_index(drop=True)
            cat_df['Forecast'] = cat_df['Balance Change'].rolling(period, min_periods=1) \
                .apply(lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True)

            last_forecast = float(cat_df['Forecast'].iloc[-1]) if not cat_df.empty else 0.0
            main_cat, sub_cat = idx
            growth_rate = float(cat_df['Balance Change'].pct_change().fillna(0).mean()) if not cat_df.empty else 0.0

            for i in range(1, forecast_months + 1):
                projected = last_forecast * (1 + growth_rate)
                forecast_list.append({
                    'Month': f'Next Month +{i}',
                    'Main Category': main_cat,
                    'Subcategory': sub_cat,
                    'Forecast': projected
                })
                last_forecast = projected

        forecast_display = pd.DataFrame(forecast_list)
        if forecast_display.empty:
            fig_forecast = px.line(title='Forecast (No Data)')
        else:
            fig_forecast = px.line(
    forecast_display,
    x='Month',
    y='Forecast',
    color='Subcategory',
    line_dash='Main Category',
    markers=True,
    title='Forecast',
    color_discrete_sequence=px.colors.qualitative.Pastel  # 🎨 soft colors
)

        st.plotly_chart(fig_forecast, use_container_width=True)
    except Exception as e:
        st.warning(f"Forecasting failed: {e}")
        fig_forecast = px.line(title='Forecast (No Data)')
        st.plotly_chart(fig_forecast, use_container_width=True)

    # Downloads
    st.download_button(
        "⬇️ Download Professional PDF Report",
        data=generate_pdf(actual_summary, cumulative, forecast_display, fig_actual, fig_cum, fig_forecast),
        file_name="budget_forecast_professional.pdf"
    )

    # Excel download for tables
    excel_buf = BytesIO()
    with pd.ExcelWriter(excel_buf, engine='xlsxwriter') as writer:
        actual_summary.to_excel(writer, sheet_name='Actual_vs_Budget', index=False)
        cumulative.to_excel(writer, sheet_name='Cumulative', index=False)
        forecast_display.to_excel(writer, sheet_name='Forecast', index=False)
    excel_buf.seek(0)
    st.download_button("⬇️ Download Tables as Excel", data=excel_buf, file_name="budget_forecast_tables.xlsx")


# --------------------- Main --------------------- #
if __name__ == "__main__":
    budget_module()
