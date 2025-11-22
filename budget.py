import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

# ---------------- Load Processed Data ---------------- #
def load_data():
    if 'processed_df' not in st.session_state:
        st.error("No processed data found. Please upload your transactions first.")
        return pd.DataFrame()

    df = st.session_state['processed_df'].copy()

    # Ensure Date exists and is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').fillna(pd.Timestamp.today())
    else:
        df['Date'] = pd.Timestamp.today()

    # Ensure essential columns exist
    for col in ['Main Category', 'Subcategory', 'Balance Change']:
        if col not in df.columns:
            df[col] = 0 if col == 'Balance Change' else 'Unknown'

    # Ensure Balance Change is numeric
    df['Balance Change'] = pd.to_numeric(df['Balance Change'], errors='coerce').fillna(0.0)

    return df

# ---------------- Export to Excel ---------------- #
def to_excel(df_dict):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in df_dict.items():
            if df is not None and not df.empty:
                df.to_excel(writer, index=False, sheet_name=sheet_name)
                worksheet = writer.sheets[sheet_name]
                for i, col in enumerate(df.columns):
                    col_width = max(df[col].astype(str).map(len).max(), len(col)) + 2
                    worksheet.set_column(i, i, col_width)
    output.seek(0)
    return output.getvalue()

# ---------------- Budget Module ---------------- #
def budget_module():
    st.title("📊 QuickBooks-Style Budgeting & Forecasting (Enhanced)")

    df = load_data()
    if df.empty:
        return

    # ---------------- Sidebar Filters ---------------- #
    st.sidebar.header("Filters")
    start_date = st.sidebar.date_input("Start Date", df['Date'].min())
    end_date = st.sidebar.date_input("End Date", df['Date'].max())
    main_categories = st.sidebar.multiselect("Main Category", df['Main Category'].unique())
    sub_categories = st.sidebar.multiselect("Subcategory", df['Subcategory'].unique())

    # Filter data
    df_filtered = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
    if main_categories:
        df_filtered = df_filtered[df_filtered['Main Category'].isin(main_categories)]
    if sub_categories:
        df_filtered = df_filtered[df_filtered['Subcategory'].isin(sub_categories)]
    if df_filtered.empty:
        st.info("No transactions found for selected filters.")
        return

    # ---------------- Normalize Expenses ---------------- #
    df_filtered['Balance Change'] = df_filtered.apply(
        lambda x: abs(x['Balance Change']) if x['Main Category'].lower() == 'expense' else x['Balance Change'], axis=1
    )

    # ---------------- Budget Input ---------------- #
    st.subheader("💰 Set / Edit Budgets per Category")
    budgeted = {}
    for idx, value in df_filtered.groupby(['Main Category', 'Subcategory'])['Balance Change'].sum().items():
        main_cat, sub_cat = idx
        budgeted[(main_cat, sub_cat)] = st.number_input(
            f"Budget for {main_cat} / {sub_cat}",
            value=0.0,  # Default to 0
            step=0.01
        )

    # ---------------- Actual vs Budget ---------------- #
    actual_summary = df_filtered.groupby(['Main Category', 'Subcategory'])['Balance Change'].sum().reset_index()
    actual_summary['Budgeted'] = actual_summary.apply(
        lambda x: budgeted.get((x['Main Category'], x['Subcategory']), 0), axis=1
    )
    actual_summary['Variance'] = actual_summary.apply(
        lambda x: (x['Balance Change'] - x['Budgeted']) if x['Main Category'].lower() != 'expense' else (x['Budgeted'] - x['Balance Change']),
        axis=1
    )
    actual_summary['Variance %'] = actual_summary.apply(
        lambda x: (x['Variance'] / x['Budgeted'] * 100) if x['Budgeted'] != 0 else 0, axis=1
    )

    # ---------------- KPIs ---------------- #
    total_actual = actual_summary['Balance Change'].sum()
    total_budget = actual_summary['Budgeted'].sum()
    total_variance = total_actual - total_budget
    st.metric("Total Actual", f"${total_actual:,.2f}")
    st.metric("Total Budget", f"${total_budget:,.2f}")
    st.metric("Total Variance", f"${total_variance:,.2f}")

    # ---------------- Conditional Formatting ---------------- #
    def highlight_variance(val):
        color = 'green' if val >= 0 else 'red'
        return f'color: {color}'

    st.subheader("📊 Budget vs Actual Summary")
    st.dataframe(actual_summary.style.applymap(highlight_variance, subset=['Variance']))

    # ---------------- Cumulative Actual vs Budget ---------------- #
    df_filtered['Month'] = df_filtered['Date'].dt.to_period('M').astype(str)
    cumulative = df_filtered.groupby(['Month', 'Main Category', 'Subcategory'])['Balance Change'].sum().reset_index()
    cumulative['Cumulative'] = cumulative.groupby(['Main Category', 'Subcategory'])['Balance Change'].cumsum()

    st.subheader("📈 Cumulative Actual vs Budget")
    fig_cum = px.line(
        cumulative,
        x='Month',
        y='Cumulative',
        color='Subcategory',
        line_dash='Main Category',
        markers=True,
        title='Cumulative Spending per Subcategory'
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # ---------------- Actual vs Budget Chart ---------------- #
    st.subheader("📊 Actual vs Budget Chart")
    fig = px.bar(
        actual_summary,
        x='Subcategory',
        y=['Balance Change', 'Budgeted'],
        color='Main Category',
        barmode='group',
        text_auto=True,
        title='Actual vs Budget per Subcategory'
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- Forecasting (Weighted Moving Average) ---------------- #
    st.subheader("🔮 Forecast Next Months (Weighted Moving Average)")
    forecast_months = st.number_input("Forecast Months", min_value=1, max_value=12, value=3)
    period = st.number_input("Moving Average Period (months)", min_value=1, max_value=12, value=3)

    forecast_display = pd.DataFrame()
    try:
        forecast_df = df_filtered.groupby(['Month', 'Main Category', 'Subcategory'])['Balance Change'].sum().reset_index()
        forecast_list = []
        for idx, cat_df in forecast_df.groupby(['Main Category', 'Subcategory']):
            cat_df = cat_df.sort_values('Month')
            cat_df['Forecast'] = cat_df['Balance Change'].rolling(period, min_periods=1).apply(
                lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True
            ).bfill()
            last_forecast = cat_df['Forecast'].iloc[-1] if not cat_df.empty else 0
            main_cat, sub_cat = idx
            for i in range(forecast_months):
                forecast_list.append({
                    'Month': f"Next Month +{i+1}",
                    'Main Category': main_cat,
                    'Subcategory': sub_cat,
                    'Forecast': last_forecast
                })
        forecast_display = pd.DataFrame(forecast_list)
        st.dataframe(forecast_display)
    except Exception as e:
        st.warning(f"Forecasting failed: {e}")

    # ---------------- Forecast Chart ---------------- #
    st.subheader("📉 Forecast Chart")
    if not forecast_display.empty:
        fig2 = px.line(
            forecast_display,
            x='Month',
            y='Forecast',
            color='Subcategory',
            line_dash='Main Category',
            markers=True,
            title='Forecast per Subcategory'
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ---------------- Download Reports ---------------- #
    df_dict = {
        'Actual vs Budget': actual_summary,
        'Cumulative Spending': cumulative,
        'Forecast': forecast_display
    }
    st.download_button(
        "⬇️ Download Budget & Forecast Report",
        data=to_excel(df_dict),
        file_name="budget_forecast_enhanced.xlsx"
    )

# ---------------- Main ---------------- #
if __name__ == "__main__":
    budget_module()
