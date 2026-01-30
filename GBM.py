import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import zscore
import io

st.set_page_config(
    page_title="GBM Multi-Omics Biomarker Finder",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME OVERRIDE FOR LIGHT MODE ---
# Injecting custom CSS to enforce a clean, high-contrast light theme
st.markdown(
    """
    <style>
    /* Main Streamlit container background (light grey) */
    .stApp {
        background-color: #F8F8F8; 
    }

    /* Sidebar background (white) */
    .stSidebar {
        background-color: #FFFFFF; 
        border-right: 1px solid #E0E0E0;
    }

    /* Headers and Text */
    h1, h2, h3, h4, .stMarkdown {
        color: #333333;
    }

    /* Primary button for a clean blue highlight */
    .stButton>button[kind="primary"] {
        color: white;
        background-color: #1E64C8;
        border-color: #1E64C8;
    }

    /* Subtle container styling for charts and dataframes */
    .stPlotlyChart, .stDataFrame {
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05); /* subtle shadow */
        padding: 10px;
        background-color: white;
    }

    /* Alert boxes for high visibility in light mode */
    .stAlert {
        border-left: 6px solid;
    }
    .stAlert.success {
        border-color: #4CAF50;
    }
    .stAlert.warning {
        border-color: #FFC107;
    }
    .stAlert.info {
        border-color: #2196F3;
    }

    /* Dialog styling */
    div[data-testid="stDialog"] {
        background-color: white;
        color: #333333;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --- Data Generation and Calculation Functions ---

@st.cache_data
def generate_mock_data():
    """Generates a mock multi-omics dataset for demonstration."""
    np.random.seed(42)
    features = [f'RNA_{i}' for i in range(150)] + \
               [f'Prot_{i}' for i in range(50)] + \
               [f'Metab_{i}' for i in range(10)]

    control_samples = [f'Control_{i}' for i in range(10)]
    gbm_samples = [f'GBM_{i}' for i in range(10)]

    data = {}
    for feature in features:
        control_expr = np.random.normal(loc=5, scale=1, size=10)
        gbm_expr = np.random.normal(loc=5, scale=1, size=10)

        if feature in ['RNA_5', 'Prot_12', 'Metab_3', 'RNA_120']:
            gbm_expr = np.random.normal(loc=10, scale=1.5, size=10)
        elif feature in ['RNA_1', 'Prot_5', 'Metab_0']:
            gbm_expr = np.random.normal(loc=2, scale=0.5, size=10)

        data[feature] = np.concatenate([control_expr, gbm_expr])

    df = pd.DataFrame(data, index=control_samples + gbm_samples).T
    df.index.name = 'Feature'
    df['Omics_Type'] = df.index.map(lambda x: x.split('_')[0])
    return df


@st.cache_data
def calculate_biomarkers(df):
    """
    Simulates the ML model output by calculating Z-scores and identifying
    high-magnitude features.
    """
    sample_df = df.drop(columns=['Omics_Type'], errors='ignore').copy()

    control_cols = [col for col in sample_df.columns if 'Control' in col]
    gbm_cols = [col for col in sample_df.columns if 'GBM' in col]

    if not control_cols or not gbm_cols:
        st.error("Data must contain columns with 'Control' and 'GBM' in their names to run the analysis.")
        return pd.DataFrame()

    for col in control_cols + gbm_cols:
        sample_df[col] = pd.to_numeric(sample_df[col], errors='coerce')

    sample_df.dropna(subset=control_cols + gbm_cols, how='all', inplace=True)

    control_mean = sample_df[control_cols].mean(axis=1)
    control_std = sample_df[control_cols].std(axis=1)

    sample_df['GBM_Mean'] = sample_df[gbm_cols].mean(axis=1)
    sample_df['Control_Mean'] = control_mean

    epsilon = 1e-6
    sample_df['Log2_Fold_Change'] = np.log2((sample_df['GBM_Mean'] + epsilon) / (sample_df['Control_Mean'] + epsilon))
    sample_df['Z_Score'] = (sample_df['GBM_Mean'] - control_mean) / (control_std + epsilon)

    np.random.seed(42)
    sample_df['-log10_P_Value'] = np.abs(sample_df['Z_Score']) * 0.8 + np.random.rand(len(sample_df)) * 0.5

    results_df = sample_df.merge(df[['Omics_Type']], left_index=True, right_index=True, how='left')

    return results_df.sort_values(by='Z_Score', ascending=False)


def create_volcano_plot(data, z_threshold, p_threshold):
    """Generates an interactive Volcano Plot using Plotly."""

    def get_status(row):
        if abs(row['Z_Score']) >= z_threshold and row['-log10_P_Value'] >= p_threshold:
            if row['Z_Score'] > 0:
                return 'Upregulated Biomarker'
            else:
                return 'Downregulated Biomarker'
        return 'Not Significant'

    data['Biomarker_Status'] = data.apply(get_status, axis=1)

    fig = px.scatter(
        data,
        x='Log2_Fold_Change',
        y='-log10_P_Value',
        color='Biomarker_Status',
        hover_data={'Feature': data.index, 'Omics_Type': True, 'Z_Score': ':.2f', 'Log2_Fold_Change': ':.2f'},
        color_discrete_map={
            'Upregulated Biomarker': '#E31B23',  # Bright Red
            'Downregulated Biomarker': '#1E64C8',  # Bright Blue
            'Not Significant': 'gray'
        },
        title='Volcano Plot: Differential Feature Expression (GBM vs Control)',
        labels={
            'Log2_Fold_Change': 'Log2(GBM Mean / Control Mean)',
            '-log10_P_Value': '-log10(P-Value Simulation)'
        }
    )

    log2_fc_approx = data[abs(data['Z_Score']) > z_threshold]['Log2_Fold_Change'].abs().mean()
    if pd.isna(log2_fc_approx) or log2_fc_approx < 0.5:
        log2_fc_approx = 1.0

    fig.add_vline(x=log2_fc_approx, line_width=1, line_dash="dash", line_color="#CC3300",
                  name=f'Log2 FC > {log2_fc_approx:.2f} (Approx)')
    fig.add_vline(x=-log2_fc_approx, line_width=1, line_dash="dash", line_color="#336699",
                  name=f'Log2 FC < -{log2_fc_approx:.2f} (Approx)')

    fig.add_hline(y=p_threshold, line_width=1, line_dash="dash", line_color="black",
                  name=f'-log10 P-Value = {p_threshold}')

    fig.update_layout(
        height=600,
        margin=dict(t=50, b=50, l=50, r=50),
        plot_bgcolor='#F7F7F7',  # Light grey plot area
        paper_bgcolor='white',  # White background around plot
        font=dict(color="#333333"),  # Dark text
        hovermode="closest",
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8), selector=dict(mode='markers'))

    return fig


def create_biomarker_heatmap(input_df, biomarkers_df):
    """Generates a heatmap for the identified biomarkers using normalized expression data."""
    biomarker_features = biomarkers_df.index.tolist()
    sample_cols = [col for col in input_df.columns if 'Control' in col or 'GBM' in col]
    raw_biomarker_data = input_df.loc[biomarker_features, sample_cols]

    if raw_biomarker_data.empty:
        return None

    try:
        normalized_data = raw_biomarker_data.apply(zscore, axis=1, result_type='broadcast')
    except Exception as e:
        st.warning(f"Normalization failed: {e}. Using raw data for heatmap.")
        normalized_data = raw_biomarker_data

    plot_data = normalized_data.T

    control_samples = [col for col in plot_data.index if 'Control' in col]
    gbm_samples = [col for col in plot_data.index if 'GBM' in col]

    ordered_index = control_samples + gbm_samples
    plot_data = plot_data.loc[ordered_index]

    fig = px.imshow(
        plot_data,
        x=plot_data.index,
        y=plot_data.columns,
        color_continuous_scale='RdBu_r',  # Light theme color scheme (red/blue reversed)
        aspect='auto',
        title='Heatmap of Normalized Expression for Identified Biomarkers',
        labels={'x': 'Sample', 'y': 'Biomarker Feature', 'color': 'Z-Score (Norm. Expression)'}
    )

    fig.update_xaxes(side="bottom", tickangle=45)

    height = min(1000, 30 * len(biomarker_features) + 200)

    fig.update_layout(
        height=height,
        margin=dict(t=50, b=100, l=50, r=50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color="#333333"),
    )

    return fig


def parse_data(source_type, data_input=None):
    """Handles parsing for all three data sources and returns the DataFrame."""
    df = None
    st.session_state['data_load_status'] = "Pending..."

    if source_type == "Use Mock Data":
        df = generate_mock_data()
        st.session_state['data_load_status'] = "success"

    elif source_type == "Upload Your CSV/TSV":
        uploaded_file = data_input
        if uploaded_file is not None:
            try:
                # Use pandas to auto-detect delimiter (sep=None, engine='python')
                df = pd.read_csv(uploaded_file, index_col=0, sep=None, engine='python')
                df.index.name = 'Feature'
                df['Omics_Type'] = df.index.map(lambda x: x.split('_')[0])
                st.session_state['data_load_status'] = "success"
            except Exception as e:
                st.session_state['data_load_status'] = f"Error loading file: {e}"
                return None

    elif source_type == "Manual Text Input":
        manual_data = data_input
        if manual_data and manual_data.strip():
            try:
                df = pd.read_csv(io.StringIO(manual_data), sep=None, engine='python', index_col=0)
                df.index.name = 'Feature'
                df['Omics_Type'] = df.index.map(lambda x: x.split('_')[0])
                st.session_state['data_load_status'] = "success"
            except Exception as e:
                st.session_state['data_load_status'] = f"Error parsing input: {e}"
                return None
        else:
            st.session_state['data_load_status'] = "Manual input is empty."
            return None

    # Final check for Control/GBM columns
    if df is not None and st.session_state['data_load_status'] == "success":
        if not any('Control' in col for col in df.columns) or not any('GBM' in col for col in df.columns):
            st.session_state['data_load_status'] = "Error: Data must contain both 'Control' and 'GBM' in sample names."
            return None

    return df


# --- Dialog Implementation ---

def data_input_dialog():
    """Renders the pop-up dialog for data input."""

    # FIX APPLIED: Removed 'as dialog' as st.dialog does not support assignment to a variable
    with st.dialog("Load Multi-Omics Data", width="large"):
        st.markdown("### Select or Input Your Multi-Omics Data")

        data_source = st.radio(
            "Select Data Source",
            ("Use Mock Data", "Upload Your CSV/TSV", "Manual Text Input"),
            key='data_source_radio'
        )

        data_input = None

        if data_source == "Upload Your CSV/TSV":
            data_input = st.file_uploader(
                "Upload Multi-Omics Data (CSV/TSV)",
                type=["csv", "tsv"],
                help="Features as rows, Samples as columns. Sample names must contain 'Control' or 'GBM'."
            )

        elif data_source == "Manual Text Input":
            data_input = st.text_area(
                "Paste Tab- or Comma-Separated Data Here",
                height=300,
                value="Feature\tControl_1\tControl_2\tGBM_1\tGBM_2\nRNA_1\t5.1\t4.9\t2.2\t2.5\nProt_12\t6.0\t5.8\t10.5\t10.2",
                help="Paste data matrix including feature names and sample headers."
            )

        st.markdown("---")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Load Data", type="primary"):
                loaded_df = parse_data(data_source, data_input)

                if loaded_df is not None and st.session_state.get('data_load_status') == "success":
                    st.session_state['input_df'] = loaded_df
                    st.session_state['data_loaded'] = True
                    st.session_state['dialog_message'] = f"Data loaded successfully from: **{data_source}**."
                    st.rerun()
                else:
                    st.error(st.session_state.get('data_load_status', "Failed to load data. Please check the format."))

        with col2:
            st.button("Cancel")

        # --- Main Application Page ---


def main_app_page():
    """Renders the core Biomarker Finder application."""
    st.title(" Glioblastoma (GBM) Multi-Omics Biomarker Identifier")
    st.markdown("---")

    # Initialize session state for the data
    if 'data_loaded' not in st.session_state:
        st.session_state['input_df'] = generate_mock_data()
        st.session_state['data_loaded'] = True

    if 'dialog_message' not in st.session_state:
        st.session_state['dialog_message'] = "Demo data loaded. Click 'Load/Change Data' to use your own data."

    # Button to trigger the data input dialog (pop-up)
    if st.button("Load/Change Data", type="secondary"):
        st.session_state['show_dialog'] = True

    if st.session_state.get('show_dialog', False):
        data_input_dialog()
        st.session_state['show_dialog'] = False  # This line is often redundant but safer if rerun doesn't happen

    # Display status message from the last dialog interaction
    if st.session_state.get('dialog_message'):
        st.info(st.session_state['dialog_message'])
        # Only clear the message if it was a success message (prevents clearing error messages before the user sees them)
        if "successfully" in st.session_state['dialog_message']:
            st.session_state['dialog_message'] = None

    input_df = st.session_state.get('input_df')

    st.sidebar.header("Analysis Settings")

    if input_df is not None and not input_df.empty:

        # 1. Thresholds (in sidebar)
        st.sidebar.markdown("---")
        st.sidebar.subheader("Biomarker Thresholds")

        z_threshold = st.sidebar.slider(
            "Absolute Z-score Threshold (Significance)",
            min_value=1.0, max_value=5.0, value=2.5, step=0.1,
            key='z_thresh',
            help="Features with |Z-score| greater than this value are considered potential biomarkers."
        )

        p_threshold = st.sidebar.slider(
            "-log10(P-Value) Threshold (Magnitude)",
            min_value=1.0, max_value=10.0, value=4.0, step=0.5,
            key='p_thresh',
            help="Features above this P-value threshold are visually highlighted."
        )

        # 2. Model Execution Button
        st.sidebar.markdown("---")
        if st.sidebar.button("Run Biomarker Analysis (Simulated ML)", type="primary"):

            with st.spinner("Calculating Z-Scores and Identifying Biomarkers..."):

                results_df = calculate_biomarkers(input_df.copy())

                if results_df.empty:
                    st.error(
                        "Analysis failed. Please check your data for valid 'Control' and 'GBM' columns and ensure expression data is numeric.")
                else:
                    st.header("1. Multi-Omics Volcano Plot Visualization")
                    volcano_fig = create_volcano_plot(results_df, z_threshold, p_threshold)
                    st.plotly_chart(volcano_fig, use_container_width=True)
                    st.caption(
                        "This plot visually represents differential expression. Features in red/blue meet both Z-score and P-value thresholds.")

                    biomarkers_df = results_df[
                        (abs(results_df['Z_Score']) >= z_threshold) &
                        (results_df['-log10_P_Value'] >= p_threshold)
                        ].sort_values(by='-log10_P_Value', ascending=False)

                    st.header("2. Top Potential GBM Biomarkers")

                    if not biomarkers_df.empty:
                        st.success(
                            f" Found **{len(biomarkers_df)}** Potential Biomarkers (Z-score ≥ {z_threshold} & -log10 P-Value ≥ {p_threshold})"
                        )

                        st.dataframe(
                            biomarkers_df[
                                ['Omics_Type', 'Log2_Fold_Change', 'Z_Score', '-log10_P_Value', 'Biomarker_Status']],
                            column_config={
                                "Omics_Type": st.column_config.TextColumn("Omics Type"),
                                "Log2_Fold_Change": st.column_config.NumberColumn("Log2 Fold Change (Effect Size)",
                                                                                  format="%.2f"),
                                "Z_Score": st.column_config.NumberColumn("Z-Score", format="%.2f"),
                                "-log10_P_Value": st.column_config.NumberColumn("-log10(P-Value)", format="%.2f"),
                                "Biomarker_Status": st.column_config.TextColumn("Status")
                            },
                            hide_index=False,
                            use_container_width=True
                        )

                        st.download_button(
                            label="Download Biomarker Table as CSV",
                            data=biomarkers_df.to_csv().encode('utf-8'),
                            file_name='gbm_biomarkers_results.csv',
                            mime='text/csv',
                        )

                        st.header("3. Heatmap Visualization of Biomarkers")
                        heatmap_fig = create_biomarker_heatmap(input_df, biomarkers_df)

                        if heatmap_fig is not None:
                            st.plotly_chart(heatmap_fig, use_container_width=True)
                            st.caption(
                                "Normalized expression heatmap. Columns (samples) are grouped by Control/GBM. Rows (features) are the identified biomarkers.")

                        else:
                            st.warning("Could not generate heatmap (e.g., insufficient data variation or features).")

                    else:
                        st.warning(
                            "No biomarkers found with the current thresholds. Try adjusting the Z-score or P-Value sliders."
                        )

                    st.header("4. Full Dataset Summary")
                    st.dataframe(results_df[['Omics_Type', 'Log2_Fold_Change', 'Z_Score', '-log10_P_Value']].head(10),
                                 use_container_width=True)

        else:
            st.info("Click 'Run Biomarker Analysis' in the sidebar to process the current data.")
            st.subheader("Data Preview (Features as Rows)")
            st.dataframe(input_df.head(), use_container_width=True)

    else:
        st.warning("Please load valid multi-omics data using the 'Load/Change Data' button to start.")


def info_page():
    """Renders the 'About This Tool' page."""
    st.title(" About the GBM Multi-Omics Biomarker Identifier")
    st.markdown("---")

    st.header("Purpose and Methodology")
    st.markdown("""
    This tool is designed to identify potential Glioblastoma (GBM) biomarkers across different omics layers (RNA, Protein, Metabolite) by comparing expression levels between GBM tumor samples and healthy control samples.
    """)

    st.subheader("Simulated Analysis Approach")
    st.info(
        " **Note:** The current version uses a simulated statistical model to demonstrate the interface.")
    st.markdown("""
    The core of the analysis, the `calculate_biomarkers` step, simulates the output of a differential expression pipeline:

    1.  **Reference Calculation:** The mean ($\mu$) and standard deviation ($\sigma$) are calculated using the **Control** samples.
    2.  **Log2 Fold Change (Log2 FC):** Calculates the expression magnitude difference between the GBM group and the Control group: $\log_2(\text{Mean}_{\text{GBM}} / \text{Mean}_{\text{Control}})$.
    3.  **Z-Score Calculation:** Measures how many standard deviations the average GBM expression is from the Control mean: 
        $$Z = \frac{\text{Mean}_{\text{GBM}} - \mu_{\text{Control}}}{\sigma_{\text{Control}}}$$
    4.  **P-Value Simulation:** A placeholder for statistical significance is generated based on the absolute Z-Score, allowing features to be plotted on the Volcano Plot.

    Features that exceed the specified Z-score and P-value thresholds are classified as potential biomarkers.
    """)

    st.header("Data Requirements")
    st.markdown("""
    The tool requires expression data in a matrix format (CSV/TSV/pasted text):
    * **Rows:** Must represent the biological features (e.g., gene names, protein IDs).
    * **Columns:** Must represent individual samples and contain only numeric data.
    * **Naming Convention:** Sample names must clearly indicate the group: names containing **'GBM'** are treated as the disease group, and names containing **'Control'** are treated as the reference group.
    """)


def run_app():
    # Navigation in the sidebar
    page = st.sidebar.radio(
        "Navigation",
        ("Biomarker Finder", "About This Tool")
    )

    # Route based on selection
    if page == "Biomarker Finder":
        main_app_page()
    elif page == "About This Tool":
        info_page()


# Execute the application
if 'input_df' not in st.session_state:
    st.session_state['input_df'] = generate_mock_data()
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = True

run_app()