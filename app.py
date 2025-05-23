import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from models import train_revenue_model, train_event_count_model, train_catering_model
from utils import load_data, preprocess_data, create_features
from weather_utils import get_weather_data # Import the new weather function
from datetime import datetime, timedelta
from openai import OpenAI # Import OpenAI library
import io # Needed for handling file bytes

# --- Initialize session state variables ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'login_error' not in st.session_state:
    st.session_state.login_error = False

# --- Login Authentication Function ---
def authenticate(username, password):
    # Hardcoded credentials as specified
    valid_credentials = {
        "haroonqureshi@omnixm.com": "CateringTest",
        "dror.rosentraub@gmail.com": "CateringTest"
    }
    return username in valid_credentials and password == valid_credentials[username]

# --- Login Page Function ---
def show_login_page():
    # Hide sidebar on login page and remove all white elements
    st.markdown("""
    <style>
    /* Hide ALL white elements - aggressive approach */
    header, iframe, [data-testid="stHeader"], [data-testid="stToolbar"], 
    [data-testid="stDecoration"], [data-testid="stStatusWidget"], 
    [data-testid="manage-app-button"], [data-testid="stAppViewBlockContainer"] > div:first-child {
        display: none !important;
        height: 0 !important;
        width: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
    }
    
    /* Force full-page purple background */
    html, body, .stApp, [data-testid="stAppViewContainer"],
    .main, [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #7D3C98 0%, #9B59B6 100%) !important;
        background-attachment: fixed !important;
    }
    
    /* Hide sidebar */
    [data-testid="stSidebar"] {display: none !important;}
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    
    /* Full-width container with no background */
    .main .block-container {
        background-color: transparent !important;
        max-width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Style login container */
    .login-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 40px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        text-align: center;
        border-top: 5px solid #9B59B6;
        max-width: 450px;
        margin: 0 auto;
        color: white !important;
    }
    
    /* White text for labels */
    .stTextInput label, h1, p, div, span {
        color: white !important;
    }
    
    /* Login button styling */
    [data-testid="stFormSubmitButton"] button {
        background: #9B59B6 !important;
        font-size: 18px !important;
        padding: 14px 24px !important;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add spacing at the top
    st.markdown("<div style='padding-top: 15vh;'></div>", unsafe_allow_html=True)
    
    # Adjust column layout for better centering
    col1, col2_content, col3 = st.columns([1, 1.5, 1]) 
    with col2_content:
        # Login container start 
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        # Header
        st.markdown('<h1 style="color: white; font-size: 28px; font-weight: 700; margin-bottom: 30px; text-align: center;">Eventions Dashboard Login</h1>', unsafe_allow_html=True)
        
        # Login Form
        with st.form("login_form"):
            username = st.text_input("Email", placeholder="Enter your email")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if authenticate(username, password):
                    st.session_state.logged_in = True
                    st.session_state.login_error = False
                    st.rerun()
                else:
                    st.session_state.login_error = True
        
        # Error message
        if st.session_state.login_error:
            st.markdown('<div style="color: #FF5555; margin-top: 15px; font-size: 14px; font-weight: 500;">Invalid email or password. Please try again.</div>', unsafe_allow_html=True)
        
        # Container end
        st.markdown('</div>', unsafe_allow_html=True)

        # Display the logo below the login container
        st.markdown("<div style='text-align: center; margin-top: 30px;'>", unsafe_allow_html=True)
        try:
            st.image("main-logo.png", width=150) # Logo below the form
        except Exception as e:
            st.warning(f"Could not load logo: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

# --- Helper function for creating styled bar charts ---
def create_bar_chart(df, y_col, x_col, title, x_label, text_template, color, full_width=False):
    if df.empty:
        st.info(f"No data to display for '{title}'.")
        return
    fig = px.bar(df, 
                 y=y_col, x=x_col, 
                 orientation='h',
                 title=title,
                 labels={x_col: x_label, y_col: ''},
                 text=x_col)
    fig.update_traces(marker_color=color, texttemplate=text_template, textposition='outside')
    fig.update_layout(height=400, yaxis_title=None, yaxis=dict(autorange="reversed"), plot_bgcolor='white', margin=dict(l=10, r=20, t=50, b=20))
    if full_width:
        st.plotly_chart(fig, use_container_width=True)
    else:
        # This else case might not be needed if we always use columns for multiple charts
        st.plotly_chart(fig, use_container_width=True) 

# --- Helper function for creating styled line charts ---
def create_line_chart(df, title):
    if df.empty:
        st.info(f"No data to display for '{title}'.")
        return
    fig = go.Figure()
    if 'ActualRevenue' in df.columns: fig.add_trace(go.Scatter(x=df['MonthYear'], y=df['ActualRevenue'], mode='lines+markers', name='Actual Revenue', line=dict(color='#FF4500')))
    if 'OrderedRevenue' in df.columns: fig.add_trace(go.Scatter(x=df['MonthYear'], y=df['OrderedRevenue'], mode='lines+markers', name='Ordered Revenue', line=dict(color='#FF6347')))
    if 'ForecastRevenue' in df.columns: fig.add_trace(go.Scatter(x=df['MonthYear'], y=df['ForecastRevenue'], mode='lines+markers', name='Forecast Revenue', line=dict(color='#FF7F50')))
    
    fig.update_layout(
        title=title,
        xaxis_title='Month',
        yaxis_title='Revenue ($',
        height=450,
        plot_bgcolor='white',
        legend_title_text='Revenue Type',
        margin=dict(l=40, r=20, t=50, b=50)
    )
    st.plotly_chart(fig, use_container_width=True)

# Set page config
st.set_page_config(
    page_title="Eventions - Enterprise Catering Dashboard",
    page_icon="🍽️",
    layout="wide"
)

# Custom CSS for purple theme
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #4A235A !important; /* Dark purple for sidebar */
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div[role="option"] /* For selectbox options */
    {
        color: white !important; /* White text for sidebar */
    }
    /* Larger title for sidebar */
    .sidebar-title {
        font-size: 32px;
        font-weight: 800;
        text-align: center;
        margin: 20px 0 30px 0;
        color: white;
    }
    .metric-card {
        background-color: white;
        border: 1px solid #DCDCDC;
        border-radius: 8px; /* Slightly more rounded */
        padding: 20px; /* More padding */
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); /* Add a subtle shadow */
    }
    .metric-card h2 {
        color: #333333;
        font-size: 32px; /* Adjusted font size */
        margin-bottom: 5px; /* Space between number and title */
    }
    .metric-card p {
        color: #555555;
        font-size: 14px;
        margin: 0;
    }
    /* Style tab headers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px; /* Taller tabs */
        white-space: pre-wrap;
        background-color: #f0f0f0;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        font-size: 16px;
        padding: 8px 24px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4A235A !important;
        color: white !important;
    }
    /* Center radio buttons */
    div[role="radiogroup"] {
        justify-content: center;
        text-align: center;
    }
    .stRadio > label > div {
        display: flex;
        justify-content: center;
        gap: 10px; /* Add gap between radio buttons */
    }
    /* For tab page title */
    .tab-page-title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 20px;
    }
    /* Improve form layout */
    .stForm [data-testid="stForm"] {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
    }
    .stForm div[data-baseweb="form-control"] {
        margin-bottom: 20px;
    }
    /* Sidebar divider */
    .sidebar-divider {
        border: 0;
        height: 1px;
        background: rgba(255, 255, 255, 0.3);
        margin: 1rem 0;
    }
    /* Footer styling */
    .sidebar-footer {
        position: fixed;
        bottom: 10px; /* Adjusted bottom padding */
        left: 0;
        width: 330px; 
        text-align: center;
        /* Removed color and font-size as it might not be needed if only logo is there */
    }
    /* CSS for st.image within the sidebar footer can be targeted if needed, 
       but st.image(width=...) is often sufficient */
</style>
""", unsafe_allow_html=True)

# --- Main App Control Flow ---
if not st.session_state.logged_in:
    show_login_page()
else:
    # --- Data Loading Function (handles both default and uploaded data) ---
    @st.cache_data
    def get_processed_data(uploaded_file_content=None, uploaded_file_name=None):
        df_loaded = None
        if uploaded_file_content is not None and uploaded_file_name is not None:
            st.sidebar.success(f"Processing uploaded file: {uploaded_file_name}")
            try:
                file_extension = uploaded_file_name.split('.')[-1].lower()
                if file_extension == 'csv':
                    df_loaded = pd.read_csv(io.BytesIO(uploaded_file_content), encoding='latin1')
                elif file_extension in ['xls', 'xlsx']:
                    df_loaded = pd.read_excel(io.BytesIO(uploaded_file_content), engine='openpyxl' if file_extension == 'xlsx' else None) # Add encoding if supported by pandas for excel, or handle differently.
                    # For Excel, pandas read_excel doesn't directly take 'encoding' in the same way as read_csv.
                    # The encoding is usually handled by the Excel engine (e.g., openpyxl, xlrd).
                    # If specific encoding issues persist with Excel, it might require a different approach,
                    # such as reading the file differently or ensuring the Excel file itself is saved with a compatible encoding.
                    # For now, assuming the default engine handles common encodings.
                    # If latin1 is strictly needed for .xls/.xlsx and causes issues, we might need to decode bytes before passing to read_excel.
                    # However, directly adding encoding='latin1' to read_excel is not a standard parameter.
                    # Let's reconsider this part if specific errors arise for Excel.
                    # For now, let's assume common Excel encodings are handled or try to decode if it's a general byte stream.
                    # A common pattern is to try decoding if it's bytes.
                    try:
                        # Attempt to read directly, most engines handle common encodings.
                        df_loaded = pd.read_excel(io.BytesIO(uploaded_file_content))
                    except UnicodeDecodeError:
                        # If a UnicodeDecodeError occurs, try decoding with latin1 then reading
                        try:
                            decoded_content = uploaded_file_content.decode('latin1')
                            df_loaded = pd.read_excel(io.StringIO(decoded_content)) # Use StringIO for decoded string
                        except Exception as decode_err:
                            st.sidebar.error(f"Error decoding Excel with latin1: {decode_err}")
                            return None
                    except Exception as e_excel: # Catch other excel reading errors
                         st.sidebar.error(f"Error reading Excel file: {e_excel}")
                         return None

                else:
                    st.sidebar.error(f"Unsupported file type: .{file_extension}. Please upload CSV or Excel.")
            except Exception as e:
                st.sidebar.error(f"Error reading uploaded file: {e}")
                return None
        else:
            # Load default data if no file is uploaded
            df_loaded = load_data() # Assumes load_data() is defined in utils.py

        if df_loaded is not None:
            try:
                df_processed = preprocess_data(df_loaded) # Assumes preprocess_data() is defined in utils.py
                df_processed['StartDate'] = pd.to_datetime(df_processed['StartDate'])
                if df_processed['StartDate'].dt.tz is None:
                    df_processed['StartDate'] = df_processed['StartDate'].dt.tz_localize('UTC')
                else:
                    df_processed['StartDate'] = df_processed['StartDate'].dt.tz_convert('UTC')
                return df_processed
            except Exception as e:
                st.error(f"Error preprocessing data: {e}. Please ensure the data structure is correct.")
                return None
        return None

    # --- Sidebar --- 
    with st.sidebar:
        st.markdown('<div class="sidebar-title">Eventions</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        st.subheader("Upload Data (Optional)")
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"], label_visibility="collapsed")
        
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.subheader("Date Range")
        # Min/max dates for date_input will be set after data is loaded

        # Placeholder for df_processed_data_global before it's loaded
        df_processed_data_global_for_sidebar = None

        if uploaded_file is not None:
            df_processed_data_global_for_sidebar = get_processed_data(uploaded_file_content=uploaded_file.getvalue(), uploaded_file_name=uploaded_file.name)
        else:
            df_processed_data_global_for_sidebar = get_processed_data() 
        
        if df_processed_data_global_for_sidebar is not None and not df_processed_data_global_for_sidebar.empty:
            min_date_sidebar = df_processed_data_global_for_sidebar['StartDate'].min().date()
            max_date_sidebar = df_processed_data_global_for_sidebar['StartDate'].max().date()
        else:
            # Fallback dates if data loading fails or df is empty, to prevent errors with date_input
            min_date_sidebar = datetime.now().date() - timedelta(days=365)
            max_date_sidebar = datetime.now().date()
            if df_processed_data_global_for_sidebar is None: # Only show error if it was a loading failure
                 st.warning("Could not load data for sidebar filters. Using default date range.")

        date_range = st.date_input(
            "Select Date Range",
            value=(min_date_sidebar, max_date_sidebar),
            min_value=min_date_sidebar,
            max_value=max_date_sidebar,
            label_visibility="collapsed"
        )
        
        # The rest of the sidebar filters will depend on df_processed_data_global being loaded
        # So, we load it definitively here for the main app logic and re-use for filters if possible

    if uploaded_file is not None:
        df_processed_data_global = get_processed_data(uploaded_file_content=uploaded_file.getvalue(), uploaded_file_name=uploaded_file.name)
    else:
        df_processed_data_global = get_processed_data()

    if df_processed_data_global is None or df_processed_data_global.empty:
        st.error("Failed to load or process data. Please check the console for errors or upload a valid file. Application cannot proceed.")
        st.stop() # Stop execution if data is not available

    # --- Re-populate sidebar filters based on the final loaded data ---
    with st.sidebar:
        # Event Type Filter (using the final df_processed_data_global)
        # st.subheader("Event Type")
        unique_event_types = sorted(df_processed_data_global['EventType_Description'].unique())
        selected_event_types = st.multiselect("Event Type", unique_event_types, default=unique_event_types)
        
        st.subheader("Catered Status")
        catered_only_filter_option = st.selectbox("Filter Catered Events", ["All Events", "Catered Only", "Non-Catered Only"], index=0, label_visibility="collapsed")
        
        # st.subheader("Business Group")
        unique_business_groups = sorted(df_processed_data_global['BusinessGroup_Description'].unique())
        selected_business_groups = st.multiselect("Business Group", unique_business_groups, default=unique_business_groups)
        
        if 'UsageName' in df_processed_data_global.columns:
            # st.subheader("Planner Name")
            unique_planners = sorted(df_processed_data_global['UsageName'].dropna().unique())
            if not unique_planners: unique_planners = ['N/A']
            selected_planners = st.multiselect("Planner Name", unique_planners, default=unique_planners)
        else:
            selected_planners = []
                
        if 'OrderedAttendance' in df_processed_data_global.columns:
            st.subheader("Event Size")
            min_att = int(df_processed_data_global['OrderedAttendance'].min())
            max_att = int(df_processed_data_global['OrderedAttendance'].max())
            if min_att == max_att: max_att +=1
            attendance_range = st.slider("Select Event Size Range", min_att, max_att, (min_att, max_att), label_visibility="collapsed")
        else:
            attendance_range = (0,0)

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.subheader("🤖 Chatbot Settings")
        if 'openai_api_key' not in st.session_state: st.session_state.openai_api_key = ""
        if 'chatbot_model' not in st.session_state: st.session_state.chatbot_model = "gpt-4o-mini"

        st.session_state.openai_api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.openai_api_key, help="Enter your OpenAI API Key to enable the chatbot.", label_visibility="collapsed", placeholder="Enter OpenAI API Key")
        st.session_state.chatbot_model = st.selectbox("Chatbot Model", options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"].index(st.session_state.chatbot_model), help="Select the OpenAI model for the chatbot.", label_visibility="collapsed")

        # Add logout button to sidebar
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

        # Footer with company logo using st.image
        # Ensure "main-logo.png" is in the same directory as app.py or provide the correct path.
        st.markdown('<div class="sidebar-footer">', unsafe_allow_html=True)
        try:
            st.image("main-logo.png", width=150) # Adjust width as needed
        except Exception as e:
            st.warning(f"Could not load logo: main-logo.png. Error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Apply Filters ---
    current_filters = filtered_df = df_processed_data_global.copy()
    if date_range:
        start_date_filter = pd.to_datetime(date_range[0]).tz_localize('UTC')
        end_date_filter = pd.to_datetime(date_range[1]).tz_localize('UTC')
        current_filters = current_filters[(current_filters['StartDate'] >= start_date_filter) & (current_filters['StartDate'] <= end_date_filter)]
    if selected_event_types:
        current_filters = current_filters[current_filters['EventType_Description'].isin(selected_event_types)]
    if catered_only_filter_option == "Catered Only":
        current_filters = current_filters[current_filters['EventType_Description'] == 'Catering Only']
    elif catered_only_filter_option == "Non-Catered Only":
        current_filters = current_filters[current_filters['EventType_Description'] != 'Catering Only']
    if selected_business_groups:
        current_filters = current_filters[current_filters['BusinessGroup_Description'].isin(selected_business_groups)]
    if 'UsageName' in current_filters.columns and selected_planners and 'N/A' not in selected_planners:
        current_filters = current_filters[current_filters['UsageName'].isin(selected_planners)]
    if 'OrderedAttendance' in current_filters.columns and attendance_range != (0,0):
        current_filters = current_filters[(current_filters['OrderedAttendance'] >= attendance_range[0]) & (current_filters['OrderedAttendance'] <= attendance_range[1])]

    # DataFrame with the first row for each event (for KPIs that need this, like revenue sums from distinct events)
    df_event_level_kpi = current_filters.drop_duplicates(subset='EventID', keep='first').copy()

    # DataFrame with max OrderedAttendance per EventID (to align with DAX logic for attendee counts)
    if 'EventID' in current_filters.columns and 'OrderedAttendance' in current_filters.columns:
        df_event_max_attendance = current_filters.groupby('EventID', as_index=False)['OrderedAttendance'].max()
        # Merge with other necessary event-level details from df_event_level_kpi (like EventType_Description, BusinessGroup_Description, StartDate, ActualRevenue)
        # This ensures we have one row per event with its MAX attendance and other consistent first-occurrence details.
        df_event_max_attendance = pd.merge(
            df_event_max_attendance, 
            df_event_level_kpi.drop(columns=['OrderedAttendance'], errors='ignore'), # Drop its own OrderedAttendance to avoid conflict
            on='EventID',
            how='left'
        )
    else:
        # Fallback if essential columns are missing, though app should stop earlier if df_processed_data_global is bad
        df_event_max_attendance = df_event_level_kpi.copy() # Will use first attendance if max cannot be calculated

    # --- Fetch and Merge Weather Data ---
    @st.cache_data
    def load_and_merge_weather_data(processed_data_df):
        if processed_data_df is None or processed_data_df.empty:
            st.warning("Main data is not available, cannot fetch weather data for merging.")
            return pd.DataFrame() 

        start_date_param = processed_data_df['StartDate'].min().date()
        end_date_param = processed_data_df['StartDate'].max().date()
        today = datetime.now().date()

        # Cap end_date_param at today
        actual_end_date = min(end_date_param, today)
        
        # Ensure start_date_param is not after actual_end_date
        actual_start_date = min(start_date_param, actual_end_date)

        # st.info(f"[Merge] Weather data request range: {actual_start_date} to {actual_end_date}") # Removed this line
        weather_df = get_weather_data(actual_start_date, actual_end_date)
        
        if weather_df.empty:
            st.warning("Could not retrieve weather data.")
            return processed_data_df # Return original data if weather data fails

        # Prepare main data for merge: ensure StartDate is just a date object
        processed_data_df_copy = processed_data_df.copy()
        processed_data_df_copy['merge_date'] = pd.to_datetime(processed_data_df_copy['StartDate']).dt.date
        
        # Merge weather data
        # The weather_df 'date' column is already datetime.date objects from the updated weather_utils.py
        merged_df = pd.merge(processed_data_df_copy, weather_df, left_on='merge_date', right_on='date', how='left')
        merged_df.drop(columns=['merge_date', 'date'], inplace=True, errors='ignore') # Clean up merge keys
        
        return merged_df

    # Load and merge weather data with the globally filtered data for initial display and general use
    df_merged_with_weather = load_and_merge_weather_data(current_filters.copy()) # Use a copy of current_filters

    # WMO Weather code descriptions (simplified)
    weather_code_descriptions = {
        0: 'Clear sky',
        1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
        45: 'Fog', 48: 'Depositing rime fog',
        51: 'Light drizzle', 53: 'Moderate drizzle', 55: 'Dense drizzle',
        56: 'Light freezing drizzle', 57: 'Dense freezing drizzle',
        61: 'Slight rain', 63: 'Moderate rain', 65: 'Heavy rain',
        66: 'Light freezing rain', 67: 'Heavy freezing rain',
        71: 'Slight snow fall', 73: 'Moderate snow fall', 75: 'Heavy snow fall',
        77: 'Snow grains',
        80: 'Slight rain showers', 81: 'Moderate rain showers', 82: 'Violent rain showers',
        85: 'Slight snow showers', 86: 'Heavy snow showers',
        95: 'Thunderstorm', # Slight or moderate
        96: 'Thunderstorm with slight hail', 99: 'Thunderstorm with heavy hail'
    }

    if not df_merged_with_weather.empty and 'weathercode' in df_merged_with_weather.columns:
        df_merged_with_weather['weather_condition'] = df_merged_with_weather['weathercode'].map(weather_code_descriptions).fillna('Unknown')

    # Create main application tabs for navigation
    tab_event_dashboard, tab_event_details, tab_attendee_dashboard, tab_revenue_dashboard, tab_advanced_analytics, tab_forecasting, tab_weather_impact = st.tabs([
        "📊 Event Dashboard", 
        "📋 Event Details", 
        "👥 Attendee Dashboard", 
        "💰 Revenue Dashboard", 
        "🔬 Advanced Analytics", 
        "📈 Forecasting",
        "🌍 Weather Impact Analysis"
    ])

    # --- DASHBOARD TAB ---
    with tab_event_dashboard:
        st.markdown('<div class="tab-page-title">Eventions Dashboard</div>', unsafe_allow_html=True)

        # --- Calculate metrics for KPI cards (using filtered data) ---
        total_events_val = df_event_level_kpi['EventID'].nunique()
        total_revenue_val = df_event_level_kpi['ActualRevenue'].sum() if 'ActualRevenue' in df_event_level_kpi.columns else 0

        catered_events_df_kpi = df_event_level_kpi[df_event_level_kpi['EventType_Description'] == 'Catering Only']
        catered_events_val = catered_events_df_kpi['EventID'].nunique()
        catered_revenue_val = catered_events_df_kpi['ActualRevenue'].sum() if 'ActualRevenue' in catered_events_df_kpi.columns else 0

        if not current_filters.empty:
            daily_event_counts = current_filters.groupby(current_filters['StartDate'].dt.date)['EventID'].nunique()
            avg_daily_events_val = int(daily_event_counts.mean()) if not daily_event_counts.empty else 0
        else:
            avg_daily_events_val = 0

        # Metric cards
        metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
        with metric_col1: st.markdown(f'''<div class="metric-card"><p>Total Events</p><h2>{total_events_val}</h2></div>''', unsafe_allow_html=True)
        with metric_col2: st.markdown(f'''<div class="metric-card"><p>Total Event Revenue</p><h2>${total_revenue_val:,.0f}</h2></div>''', unsafe_allow_html=True)
        with metric_col3: st.markdown(f'''<div class="metric-card"><p>Catered Events</p><h2>{catered_events_val}</h2></div>''', unsafe_allow_html=True)
        with metric_col4: st.markdown(f'''<div class="metric-card"><p>Catered Events Revenue</p><h2>${catered_revenue_val:,.0f}</h2></div>''', unsafe_allow_html=True)
        with metric_col5: st.markdown(f'''<div class="metric-card"><p>Average Daily Events</p><h2>{avg_daily_events_val}</h2></div>''', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True) # Add some space

        # Time frequency selector for charts
        time_frequency_charts = st.radio("", ["Daily", "Weekly", "Monthly"], horizontal=True, index=0, key="time_freq_charts_radio", label_visibility="collapsed")

        # --- Prepare data for charts based on time_frequency_charts ---
        chart_df = current_filters.copy()
        chart_df['StartDate'] = pd.to_datetime(chart_df['StartDate'])

        if time_frequency_charts == "Daily":
            chart_df['TimeGroup'] = chart_df['StartDate'].dt.date
            chart_xaxis_title = "Date"
        elif time_frequency_charts == "Weekly":
            chart_df['TimeGroup'] = chart_df['StartDate'].dt.to_period('W').apply(lambda r: r.start_time)
            chart_xaxis_title = "Week Starting"
        else:  # Monthly
            chart_df['TimeGroup'] = chart_df['StartDate'].dt.to_period('M').apply(lambda r: r.start_time)
            chart_xaxis_title = "Month"

        # Events by time period
        events_by_time_chart = chart_df.groupby('TimeGroup')['EventID'].nunique().reset_index(name='Events')

        # Caterings by time period
        caterings_by_time_chart = chart_df[chart_df['EventType_Description'] == 'Catering Only'].groupby('TimeGroup')['EventID'].nunique().reset_index(name='CateredEvents')

        # Revenue by time period - using df_event_level for revenue sums to avoid duplication for revenue calculations
        df_event_level_charts = current_filters.drop_duplicates(subset='EventID', keep='first').copy()
        df_event_level_charts['StartDate'] = pd.to_datetime(df_event_level_charts['StartDate'])

        if time_frequency_charts == "Daily":
            df_event_level_charts['TimeGroup'] = df_event_level_charts['StartDate'].dt.date
        elif time_frequency_charts == "Weekly":
            df_event_level_charts['TimeGroup'] = df_event_level_charts['StartDate'].dt.to_period('W').apply(lambda r: r.start_time)
        else: # Monthly
            df_event_level_charts['TimeGroup'] = df_event_level_charts['StartDate'].dt.to_period('M').apply(lambda r: r.start_time)

        event_revenue_by_time_chart = df_event_level_charts.groupby('TimeGroup')['ActualRevenue'].sum().reset_index(name='Revenue')
        catering_revenue_by_time_chart = df_event_level_charts[df_event_level_charts['EventType_Description'] == 'Catering Only'].groupby('TimeGroup')['ActualRevenue'].sum().reset_index(name='CateredRevenue')

        # Attendees by event type
        if 'OrderedAttendance' in current_filters.columns and not current_filters.empty:
            attendees_by_event_type_chart = current_filters.groupby('EventType_Description')['OrderedAttendance'].mean().reset_index()
            attendees_by_event_type_chart = attendees_by_event_type_chart.sort_values('OrderedAttendance', ascending=False)
        else:
            attendees_by_event_type_chart = pd.DataFrame(columns=['EventType_Description', 'OrderedAttendance'])

        # --- Visualizations ---
        chart_row1_col1, chart_row1_col2, chart_row1_col3 = st.columns([2,2,3])

        with chart_row1_col1:
            fig_events = px.bar(events_by_time_chart, x='TimeGroup', y='Events', labels={'TimeGroup': chart_xaxis_title, 'Events': 'Events'}, title=f'Events by {time_frequency_charts}')
            fig_events.update_traces(marker_color='rgb(63, 40, 102)')
            fig_events.update_layout(height=350, margin=dict(l=40, r=20, t=50, b=40), title_font=dict(size=16), plot_bgcolor='white')
            st.plotly_chart(fig_events, use_container_width=True)

        with chart_row1_col2:
            fig_caterings = px.bar(caterings_by_time_chart, x='TimeGroup', y='CateredEvents', labels={'TimeGroup': chart_xaxis_title, 'CateredEvents': 'Caterings'}, title=f'Caterings by {time_frequency_charts}')
            fig_caterings.update_traces(marker_color='rgb(85, 57, 130)')
            fig_caterings.update_layout(height=350, margin=dict(l=40, r=20, t=50, b=40), title_font=dict(size=16), plot_bgcolor='white')
            st.plotly_chart(fig_caterings, use_container_width=True)

        with chart_row1_col3:
            if not attendees_by_event_type_chart.empty:
                fig_attendees = px.bar(attendees_by_event_type_chart.head(10), y='EventType_Description', x='OrderedAttendance', orientation='h', labels={'EventType_Description': '', 'OrderedAttendance': 'Avg Attendees'}, title='Avg Attendees Per Event (Top 10)')
                fig_attendees.update_traces(marker_color='rgb(134, 96, 208)')
                fig_attendees.update_layout(height=350, margin=dict(l=10, r=20, t=50, b=20), title_font=dict(size=16), yaxis_title=None, plot_bgcolor='white', yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_attendees, use_container_width=True)
            else:
                st.info("No attendance data for selected filters.")

        chart_row2_col1, chart_row2_col2 = st.columns(2)

        with chart_row2_col1:
            if not event_revenue_by_time_chart.empty:
                fig_event_revenue = px.line(event_revenue_by_time_chart, x='TimeGroup', y='Revenue', labels={'TimeGroup': chart_xaxis_title, 'Revenue': 'Revenue ($)'},
                                        title=f'Event Revenue by {time_frequency_charts}')
                fig_event_revenue.update_traces(line=dict(color='rgb(63, 40, 102)', width=2))
                fig_event_revenue.update_layout(height=350, margin=dict(l=40, r=20, t=50, b=40), title_font=dict(size=16), plot_bgcolor='white')
                st.plotly_chart(fig_event_revenue, use_container_width=True)
            else:
                st.info("No event revenue data for selected filters.")

        with chart_row2_col2:
            if not catering_revenue_by_time_chart.empty:
                fig_catering_revenue = px.line(catering_revenue_by_time_chart, x='TimeGroup', y='CateredRevenue', labels={'TimeGroup': chart_xaxis_title, 'CateredRevenue': 'Revenue ($)'},
                                            title=f'Catering Revenue by {time_frequency_charts}')
                fig_catering_revenue.update_traces(line=dict(color='rgb(85, 57, 130)', width=2))
                fig_catering_revenue.update_layout(height=350, margin=dict(l=40, r=20, t=50, b=40), title_font=dict(size=16), plot_bgcolor='white')
                st.plotly_chart(fig_catering_revenue, use_container_width=True)
            else:
                st.info("No catering revenue data for selected filters.")

    # --- FORECASTING TAB ---
    with tab_forecasting:
        st.markdown('<div class="tab-page-title">Advanced Forecasting Dashboard</div>', unsafe_allow_html=True)

        # Initialize session state for storing forecast results if not already present
        if 'revenue_predictions' not in st.session_state:
            st.session_state.revenue_predictions = None
        if 'event_predictions' not in st.session_state:
            st.session_state.event_predictions = None
        if 'catering_predictions' not in st.session_state:
            st.session_state.catering_predictions = None
        if 'forecast_fig' not in st.session_state:
            st.session_state.forecast_fig = None
        if 'forecast_cutoff_display' not in st.session_state:
            st.session_state.forecast_cutoff_display = None
        if 'revenue_df_full_history_display' not in st.session_state:
            st.session_state.revenue_df_full_history_display = None
        if 'event_df_full_history_display' not in st.session_state:
            st.session_state.event_df_full_history_display = None
        if 'catering_features_full_history_display' not in st.session_state:
            st.session_state.catering_features_full_history_display = None

        # --- Input Form ---
        with st.form(key='forecasting_form'):
            st.subheader("Set Forecast Parameters")
            min_date_from_data = pd.to_datetime(df_processed_data_global['StartDate']).min().date()
            max_date_from_data = pd.to_datetime(df_processed_data_global['StartDate']).max().date()

            col1, col2, col3 = st.columns(3) # Added a third column for the new radio button
            with col1:
                default_forecast_start_val = pd.Timestamp('2024-11-01', tz='UTC')
                if default_forecast_start_val.date() > max_date_from_data:
                    default_forecast_start_val = pd.Timestamp(max_date_from_data, tz='UTC')
                elif default_forecast_start_val.date() < min_date_from_data:
                    default_forecast_start_val = pd.Timestamp(min_date_from_data, tz='UTC')

                forecast_start_input = st.date_input(
                    "Forecast Starting From",
                    value=st.session_state.get('forecast_start_input_val', default_forecast_start_val.date()),
                    min_value=min_date_from_data,
                    max_value=max_date_from_data,
                    key="forecast_start_date_input_form"
                )
            with col2:
                forecast_horizon_input = st.slider(
                    "Forecast Horizon (Months)",
                    min_value=1, max_value=12, 
                    value=st.session_state.get('forecast_horizon_input_val', 4),
                    key="forecast_horizon_input_form"
                )
            with col3: # New column for forecast mode
                forecast_mode_input = st.radio(
                    "Forecast Mode",
                    options=["Lower (Fastest, RF)", "Standard (XGBoost, CIs)", "Advanced (XGBoost, Optuna, CIs)"],
                    index=0, # Default to Lower (RF)
                    key="forecast_mode_input_form",
                    help="Lower: RandomForest (fastest, no CIs). Standard: XGBoost defaults + CIs. Advanced: XGBoost + Optuna + CIs (slowest)."
                )
            
            submit_button = st.form_submit_button(label="Generate Forecasts")

        if submit_button:
            st.session_state.forecast_start_input_val = forecast_start_input
            st.session_state.forecast_horizon_input_val = forecast_horizon_input
            st.session_state.forecast_mode_input_val = forecast_mode_input # Save the mode

            if forecast_mode_input == "Lower (Fastest, RF)":
                selected_forecast_mode = "lower_rf"
            elif forecast_mode_input == "Standard (XGBoost, CIs)":
                selected_forecast_mode = "standard_xgb"
            else: # Advanced (XGBoost, Optuna, CIs)
                selected_forecast_mode = "advanced_xgb"

            forecast_start_ts = pd.Timestamp(forecast_start_input, tz='UTC')
            forecast_cutoff = pd.Timestamp(forecast_start_ts.year, forecast_start_ts.month, 1, tz='UTC')
            st.session_state.forecast_cutoff_display = forecast_cutoff

            current_date = pd.Timestamp.now(tz='UTC')

            # Revenue Forecasting Data Preparation
            event_level_df_revenue = df_processed_data_global.groupby('EventID').agg({
                'StartDate': 'first',
                'ActualRevenue': 'first', # Added missing value for the key
            }).reset_index()
            event_level_df_revenue['Month'] = pd.to_datetime(event_level_df_revenue['StartDate']).dt.to_period('M')
            revenue_per_month = event_level_df_revenue.groupby('Month')[['ActualRevenue']].sum().reset_index()
            revenue_per_month['Month'] = revenue_per_month['Month'].dt.to_timestamp()
            if revenue_per_month['Month'].dt.tz is None: revenue_per_month['Month'] = revenue_per_month['Month'].dt.tz_localize('UTC')
            else: revenue_per_month['Month'] = revenue_per_month['Month'].dt.tz_convert('UTC')
            for idx, row in revenue_per_month.iterrows():
                if row['Month'] > current_date and row['ActualRevenue'] == 0:
                    revenue_per_month.at[idx, 'ActualRevenue'] = np.nan
            st.session_state.revenue_df_full_history_display = create_features(revenue_per_month.copy(), 'ActualRevenue', 'Month')
            revenue_training_data = st.session_state.revenue_df_full_history_display[st.session_state.revenue_df_full_history_display['Month'] < forecast_cutoff].copy()
            st.session_state.revenue_predictions = train_revenue_model(revenue_training_data, forecast_horizon_input, forecast_mode=selected_forecast_mode)

            # Event Count Forecasting Data Preparation
            df_events = df_processed_data_global.copy()
            df_events['MonthPeriod'] = pd.to_datetime(df_events['StartDate']).dt.to_period('M')
            events_per_month = df_events.groupby('MonthPeriod')['EventID'].nunique().reset_index(name='Event_Count')
            events_per_month.rename(columns={'MonthPeriod': 'Month'}, inplace=True)
            events_per_month['Month'] = events_per_month['Month'].dt.to_timestamp()
            if events_per_month['Month'].dt.tz is None: events_per_month['Month'] = events_per_month['Month'].dt.tz_localize('UTC')
            else: events_per_month['Month'] = events_per_month['Month'].dt.tz_convert('UTC')
            for idx, row in events_per_month.iterrows():
                if row['Month'] > current_date and row['Event_Count'] == 0:
                    events_per_month.at[idx, 'Event_Count'] = np.nan
            st.session_state.event_df_full_history_display = create_features(events_per_month.copy(), 'Event_Count', 'Month')
            event_training_data = st.session_state.event_df_full_history_display[st.session_state.event_df_full_history_display['Month'] < forecast_cutoff].copy()
            st.session_state.event_predictions = train_event_count_model(event_training_data, forecast_horizon_input, forecast_mode=selected_forecast_mode)

            # Catering Event Forecasting Data Preparation
            catering_df_raw = df_processed_data_global[df_processed_data_global['EventType_Description'] == 'Catering Only'].copy()
            catering_df_raw['MonthPeriod'] = pd.to_datetime(catering_df_raw['StartDate']).dt.to_period('M')
            catering_per_month = catering_df_raw.groupby('MonthPeriod')['EventID'].nunique().reset_index(name='Event_Count')
            catering_per_month.rename(columns={'MonthPeriod': 'Month'}, inplace=True)
            catering_per_month['Month'] = catering_per_month['Month'].dt.to_timestamp()
            if catering_per_month['Month'].dt.tz is None: catering_per_month['Month'] = catering_per_month['Month'].dt.tz_localize('UTC')
            else: catering_per_month['Month'] = catering_per_month['Month'].dt.tz_convert('UTC')
            for idx, row in catering_per_month.iterrows():
                if row['Month'] > current_date and row['Event_Count'] == 0:
                    catering_per_month.at[idx, 'Event_Count'] = np.nan
            st.session_state.catering_features_full_history_display = create_features(catering_per_month.copy(), 'Event_Count', 'Month')
            catering_training_data = st.session_state.catering_features_full_history_display[st.session_state.catering_features_full_history_display['Month'] < forecast_cutoff].copy()
            st.session_state.catering_predictions = train_catering_model(catering_training_data, forecast_horizon_input, forecast_mode=selected_forecast_mode)

            # Show a progress indicator while models train
            with st.spinner('Training forecasting models and generating predictions...'):
                # Create Plotly Figure
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=("Revenue Forecasting", "Event Count Forecasting", "Catering Event Forecasting"),
                    vertical_spacing=0.1, specs=[[{"type": "scatter"}], [{"type": "scatter"}], [{"type": "scatter"}]]
                )
                
                # Define new color scheme for forecasting plots
                color_revenue_actual = "#FFD700"  # Gold
                color_revenue_forecast = "#FFA500"  # Orange
                color_revenue_ci = "rgba(255, 165, 0, 0.15)" # Lighter Orange CI

                color_event_actual = "#4682B4"  # SteelBlue
                color_event_forecast = "#5F9EA0"  # CadetBlue
                color_event_ci = "rgba(95, 158, 160, 0.15)" # Lighter CadetBlue CI

                color_catering_actual = "#32CD32"  # LimeGreen
                color_catering_forecast = "#2E8B57"  # SeaGreen
                color_catering_ci = "rgba(46, 139, 87, 0.15)" # Lighter SeaGreen CI

                # Revenue Plot
                if st.session_state.revenue_df_full_history_display is not None and st.session_state.revenue_predictions is not None:
                    fig.add_trace(go.Scatter(x=st.session_state.revenue_df_full_history_display['Month'], y=st.session_state.revenue_df_full_history_display['ActualRevenue'], name='Actual Revenue', line=dict(color=color_revenue_actual, width=3), legendgroup='revenue', showlegend=False), row=1, col=1)
                    fig.add_trace(go.Scatter(x=st.session_state.revenue_predictions['Month'], y=st.session_state.revenue_predictions['ForecastedRevenue'], name='Forecasted Revenue', line=dict(color=color_revenue_forecast, width=3, dash='solid'), legendgroup='revenue', showlegend=False), row=1, col=1)
                    if not st.session_state.revenue_predictions['Lower_CI'].isna().all():
                        fig.add_trace(go.Scatter(x=st.session_state.revenue_predictions['Month'].tolist() + st.session_state.revenue_predictions['Month'].tolist()[::-1], y=st.session_state.revenue_predictions['Upper_CI'].tolist() + st.session_state.revenue_predictions['Lower_CI'].tolist()[::-1], fill='toself', fillcolor=color_revenue_ci, line=dict(color='rgba(0,0,0,0)'), name='Revenue CI (90%)', legendgroup='revenue', showlegend=False), row=1, col=1)
                    fig.add_vline(x=int(st.session_state.forecast_cutoff_display.timestamp() * 1000), line_width=2, line_dash="dash", line_color="gray", row=1, col=1)
                
                # Event Plot
                if st.session_state.event_df_full_history_display is not None and st.session_state.event_predictions is not None:
                    fig.add_trace(go.Scatter(x=st.session_state.event_df_full_history_display['Month'], y=st.session_state.event_df_full_history_display['Event_Count'], name='Actual Events', line=dict(color=color_event_actual, width=3), legendgroup='events', showlegend=False), row=2, col=1)
                    fig.add_trace(go.Scatter(x=st.session_state.event_predictions['Month'], y=st.session_state.event_predictions['ForecastedRevenue'], name='Forecasted Events', line=dict(color=color_event_forecast, width=3, dash='solid'), legendgroup='events', showlegend=False), row=2, col=1)
                    if not st.session_state.event_predictions['Lower_CI'].isna().all():
                        fig.add_trace(go.Scatter(x=st.session_state.event_predictions['Month'].tolist() + st.session_state.event_predictions['Month'].tolist()[::-1], y=st.session_state.event_predictions['Upper_CI'].tolist() + st.session_state.event_predictions['Lower_CI'].tolist()[::-1], fill='toself', fillcolor=color_event_ci, line=dict(color='rgba(0,0,0,0)'), name='Events CI (90%)', legendgroup='events', showlegend=False), row=2, col=1)
                    fig.add_vline(x=int(st.session_state.forecast_cutoff_display.timestamp() * 1000), line_width=2, line_dash="dash", line_color="gray", row=2, col=1)

                # Catering Plot
                if st.session_state.catering_features_full_history_display is not None and st.session_state.catering_predictions is not None:
                    fig.add_trace(go.Scatter(x=st.session_state.catering_features_full_history_display['Month'], y=st.session_state.catering_features_full_history_display['Event_Count'], name='Actual Catering Events', line=dict(color=color_catering_actual, width=3), legendgroup='catering', showlegend=False), row=3, col=1)
                    fig.add_trace(go.Scatter(x=st.session_state.catering_predictions['Month'], y=st.session_state.catering_predictions['ForecastedRevenue'], name='Forecasted Catering Events', line=dict(color=color_catering_forecast, width=3, dash='solid'), legendgroup='catering', showlegend=False), row=3, col=1)
                    if not st.session_state.catering_predictions['Lower_CI'].isna().all():
                        fig.add_trace(go.Scatter(x=st.session_state.catering_predictions['Month'].tolist() + st.session_state.catering_predictions['Month'].tolist()[::-1], y=st.session_state.catering_predictions['Upper_CI'].tolist() + st.session_state.catering_predictions['Lower_CI'].tolist()[::-1], fill='toself', fillcolor=color_catering_ci, line=dict(color='rgba(0,0,0,0)'), name='Catering CI (90%)', legendgroup='catering', showlegend=False), row=3, col=1)
                    fig.add_vline(x=int(st.session_state.forecast_cutoff_display.timestamp() * 1000), line_width=2, line_dash="dash", line_color="gray", annotation_text="Forecast Start", annotation_position="top right", row=3, col=1)
                
                fig.update_layout(
                    height=900, 
                    title_text='Enterprise Catering Forecasts', 
                    title_font=dict(size=24),
                    showlegend=False, # Global legend removal for the figure
                    hovermode="x unified", 
                    hoverlabel=dict(bgcolor="white", font_size=12),
                    margin=dict(l=20, r=20, t=80, b=20),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                )
                fig.update_yaxes(title_text="Revenue ($)", gridcolor='rgba(0,0,0,0.1)', row=1, col=1)
                fig.update_yaxes(title_text="Number of Events", gridcolor='rgba(0,0,0,0.1)', row=2, col=1)
                fig.update_yaxes(title_text="Number of Catering Events", gridcolor='rgba(0,0,0,0.1)', row=3, col=1)
                fig.update_xaxes(title_text="", gridcolor='rgba(0,0,0,0.1)', row=1, col=1, matches='x3')
                fig.update_xaxes(title_text="", gridcolor='rgba(0,0,0,0.1)', row=2, col=1, matches='x3')
                fig.update_xaxes(title_text="Date", gridcolor='rgba(0,0,0,0.1)', row=3, col=1)
                
                st.session_state.forecast_fig = fig

            st.success("Forecast generated successfully!")

        # Display Area (always tries to display from session state)
        if st.session_state.forecast_fig:
            st.plotly_chart(st.session_state.forecast_fig, use_container_width=True)

            forecast_tabs = st.tabs(["Revenue Forecast Data", "Event Forecast Data", "Catering Forecast Data"])
            with forecast_tabs[0]:
                st.subheader("Revenue Forecast Data")
                if st.session_state.revenue_predictions is not None:
                    revenue_df_display = st.session_state.revenue_predictions.copy()
                    revenue_df_display['Month'] = revenue_df_display['Month'].dt.strftime('%Y-%m')
                    st.dataframe(revenue_df_display, use_container_width=True)
                    if st.session_state.revenue_predictions['Lower_CI'].isna().all() or st.session_state.revenue_predictions['Upper_CI'].isna().all():
                        st.info("Confidence Intervals for Revenue could not be generated due to insufficient historical data.")
                else:
                    st.info("Click 'Generate Forecasts' to see data.")

            with forecast_tabs[1]:
                st.subheader("Event Forecast Data")
                if st.session_state.event_predictions is not None:
                    event_df_display = st.session_state.event_predictions.copy()
                    event_df_display['Month'] = event_df_display['Month'].dt.strftime('%Y-%m')
                    st.dataframe(event_df_display, use_container_width=True)
                    if st.session_state.event_predictions['Lower_CI'].isna().all() or st.session_state.event_predictions['Upper_CI'].isna().all():
                        st.info("Confidence Intervals for Event Counts could not be generated due to insufficient historical data.")
                else:
                    st.info("Click 'Generate Forecasts' to see data.")

            with forecast_tabs[2]:
                st.subheader("Catering Forecast Data")
                if st.session_state.catering_predictions is not None:
                    catering_df_display = st.session_state.catering_predictions.copy()
                    catering_df_display['Month'] = catering_df_display['Month'].dt.strftime('%Y-%m')
                    st.dataframe(catering_df_display, use_container_width=True)
                    if st.session_state.catering_predictions['Lower_CI'].isna().all() or st.session_state.catering_predictions['Upper_CI'].isna().all():
                        st.info("Confidence Intervals for Catering Events could not be generated due to insufficient historical data.")
                else:
                    st.info("Click 'Generate Forecasts' to see data.")
        else:
            st.info("Please set your forecast parameters above and click 'Generate Forecasts'.")

        # --- Chatbot Section for Forecasting Tab ---
        st.markdown("---")
        st.subheader("💬 Chat with Forecasting Data")

        if "forecasting_messages" not in st.session_state:
            st.session_state.forecasting_messages = []

        # Display existing messages
        for message in st.session_state.forecasting_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input for forecasting
        if prompt_forecast := st.chat_input("Ask about the generated forecast data..."):
            if not st.session_state.openai_api_key:
                st.error("Please enter your OpenAI API Key in the sidebar to use the chatbot.")
            elif not st.session_state.get('forecast_fig') or \
                 st.session_state.get('revenue_predictions') is None or \
                 st.session_state.get('event_predictions') is None or \
                 st.session_state.get('catering_predictions') is None:
                st.warning("Please generate forecasts first to enable the chatbot for this tab.")
            else:
                st.session_state.forecasting_messages.append({"role": "user", "content": prompt_forecast})
                with st.chat_message("user"):
                    st.markdown(prompt_forecast)

                with st.spinner("Preparing data and thinking..."):
                    # Prepare Revenue Data for LLM
                    df_hist_rev = pd.DataFrame()
                    if st.session_state.get('revenue_df_full_history_display') is not None and st.session_state.get('forecast_cutoff_display') is not None:
                        df_hist_rev = st.session_state.revenue_df_full_history_display[
                            st.session_state.revenue_df_full_history_display['Month'] < st.session_state.forecast_cutoff_display
                        ][['Month', 'ActualRevenue']].copy()
                    
                    df_fcst_rev = st.session_state.revenue_predictions[['Month', 'ForecastedRevenue', 'Lower_CI', 'Upper_CI']].copy()
                    df_rev_for_llm = pd.merge(df_hist_rev, df_fcst_rev, on='Month', how='outer')
                    df_rev_for_llm.rename(columns={'ForecastedRevenue': 'Forecasted Revenue'}, inplace=True)
                    df_rev_for_llm['Month'] = pd.to_datetime(df_rev_for_llm['Month']).dt.strftime('%Y-%m-%d')
                    revenue_data_csv = df_rev_for_llm.to_csv(index=False)

                    # Prepare Event Count Data for LLM
                    df_hist_event = pd.DataFrame()
                    if st.session_state.get('event_df_full_history_display') is not None and st.session_state.get('forecast_cutoff_display') is not None:
                        df_hist_event = st.session_state.event_df_full_history_display[
                            st.session_state.event_df_full_history_display['Month'] < st.session_state.forecast_cutoff_display
                        ][['Month', 'Event_Count']].copy()
                    
                    df_fcst_event = st.session_state.event_predictions[['Month', 'ForecastedRevenue', 'Lower_CI', 'Upper_CI']].copy()
                    df_event_for_llm = pd.merge(df_hist_event, df_fcst_event, on='Month', how='outer')
                    df_event_for_llm.rename(columns={'Event_Count': 'Actual Event Count', 'ForecastedRevenue': 'Forecasted Event Count'}, inplace=True)
                    df_event_for_llm['Month'] = pd.to_datetime(df_event_for_llm['Month']).dt.strftime('%Y-%m-%d')
                    event_data_csv = df_event_for_llm.to_csv(index=False)
                    
                    # Prepare Catering Event Data for LLM
                    df_hist_catering = pd.DataFrame()
                    if st.session_state.get('catering_features_full_history_display') is not None and st.session_state.get('forecast_cutoff_display') is not None:
                        df_hist_catering = st.session_state.catering_features_full_history_display[
                            st.session_state.catering_features_full_history_display['Month'] < st.session_state.forecast_cutoff_display
                        ][['Month', 'Event_Count']].copy()

                    df_fcst_catering = st.session_state.catering_predictions[['Month', 'ForecastedRevenue', 'Lower_CI', 'Upper_CI']].copy()
                    df_catering_for_llm = pd.merge(df_hist_catering, df_fcst_catering, on='Month', how='outer')
                    df_catering_for_llm.rename(columns={'Event_Count': 'Actual Catering Event Count', 'ForecastedRevenue': 'Forecasted Catering Event Count'}, inplace=True)
                    df_catering_for_llm['Month'] = pd.to_datetime(df_catering_for_llm['Month']).dt.strftime('%Y-%m-%d')
                    catering_data_csv = df_catering_for_llm.to_csv(index=False)

                    system_prompt_forecast_chat = f'''You are an AI assistant specialized in analyzing time series forecast data.
The user has provided three datasets related to business forecasting: Revenue, Event Counts, and Catering Event Counts.
Each dataset is in CSV format and includes:
- 'Month': The month of the data.
- An 'Actual' column (e.g., 'ActualRevenue', 'Actual Event Count', 'Actual Catering Event Count') for historical data.
- A 'Forecasted' column (e.g., 'Forecasted Revenue', 'Forecasted Event Count', 'Forecasted Catering Event Count') for predicted values.
- 'Lower_CI' and 'Upper_CI': Representing the 90% confidence interval for the forecasts.

Your task is to answer questions based ONLY on this provided data.
Analyze trends, compare actuals vs. forecasts, discuss confidence intervals, and provide insights drawn directly from the numbers.
If the data does not contain the answer, or if a forecast has not been generated for a particular metric, explicitly state that.
Do not make assumptions or use external knowledge.
Today's date is {datetime.now().strftime('%Y-%m-%d')}.

Provided Revenue Data (CSV):
{revenue_data_csv}

Provided Event Count Data (CSV):
{event_data_csv}

Provided Catering Event Count Data (CSV):
{catering_data_csv}
'''
                    messages_for_api = [
                        {"role": "system", "content": system_prompt_forecast_chat},
                        {"role": "user", "content": prompt_forecast}
                    ]

                    try:
                        client = OpenAI(api_key=st.session_state.openai_api_key)
                        with st.chat_message("assistant"):
                            placeholder = st.empty() # Create a placeholder
                            response_content_stream = ""
                            for chunk in client.chat.completions.create(
                                model=st.session_state.chatbot_model,
                                messages=messages_for_api,
                                temperature=0.3,
                                stream=True,
                            ):
                                if chunk.choices[0].delta.content is not None:
                                    response_content_stream += chunk.choices[0].delta.content
                                    placeholder.markdown(response_content_stream + "▌") # Update the placeholder
                            placeholder.markdown(response_content_stream) # Display final response in the placeholder
                        st.session_state.forecasting_messages.append({"role": "assistant", "content": response_content_stream})
                    except Exception as e:
                        st.error(f"Error communicating with OpenAI: {e}")

    # --- EVENT DETAILS TAB ---
    with tab_event_details:
        st.markdown('<div class="tab-page-title">Event and Catering Details</div>', unsafe_allow_html=True)
        
        # Prepare the event details dataframe from current_filters (globally filtered data)
        event_details_base = current_filters.copy()
        event_details_for_display = event_details_base.copy() # Start with a full copy, will be filtered by in-tab selections

        # --- In-tab filters ---
        st.markdown("##### Further Filter Event Details:")
        
        # Row 1 of filters: Event Type (existing) and Planner
        filter_row1_col1, filter_row1_col2 = st.columns(2)
        with filter_row1_col1:
            if not event_details_base.empty and 'EventType_Description' in event_details_base.columns:
                unique_event_types_in_tab = ["All Event Types"] + sorted(event_details_base['EventType_Description'].unique())
                selected_event_type_in_tab = st.selectbox(
                    "Filter by Event Type:",
                    options=unique_event_types_in_tab,
                    index=0,
                    key="event_type_filter_selectbox_tab3"
                )
                if selected_event_type_in_tab != "All Event Types":
                    event_details_for_display = event_details_for_display[event_details_for_display['EventType_Description'] == selected_event_type_in_tab]
            else:
                st.selectbox("Filter by Event Type:", ["No Event Types available"], disabled=True)

        with filter_row1_col2:
            if not event_details_base.empty and 'PlannerName_Masked' in event_details_base.columns:
                unique_planners = ["All Planners"] + sorted(event_details_base['PlannerName_Masked'].dropna().unique())
                selected_planner = st.selectbox(
                    "Filter by Planner:",
                    options=unique_planners,
                    index=0,
                    key="planner_filter_selectbox_tab3"
                )
                if selected_planner != "All Planners":
                    event_details_for_display = event_details_for_display[event_details_for_display['PlannerName_Masked'] == selected_planner]
            else:
                st.selectbox("Filter by Planner:", ["No Planners available"], disabled=True)

        # Row 2 of filters: Business Group and Space
        filter_row2_col1, filter_row2_col2 = st.columns(2)
        with filter_row2_col1:
            if not event_details_base.empty and 'BusinessGroup_Description' in event_details_base.columns:
                unique_biz_groups = ["All Business Groups"] + sorted(event_details_base['BusinessGroup_Description'].dropna().unique())
                selected_biz_group = st.selectbox(
                    "Filter by Business Group:",
                    options=unique_biz_groups,
                    index=0,
                    key="biz_group_filter_selectbox_tab3"
                )
                if selected_biz_group != "All Business Groups":
                    event_details_for_display = event_details_for_display[event_details_for_display['BusinessGroup_Description'] == selected_biz_group]
            else:
                st.selectbox("Filter by Business Group:", ["No Business Groups available"], disabled=True)

        with filter_row2_col2:
            if not event_details_base.empty and 'Space_Description' in event_details_base.columns:
                unique_spaces = ["All Spaces"] + sorted(event_details_base['Space_Description'].dropna().unique())
                selected_space = st.selectbox(
                    "Filter by Space:",
                    options=unique_spaces,
                    index=0,
                    key="space_filter_selectbox_tab3"
                )
                if selected_space != "All Spaces":
                    event_details_for_display = event_details_for_display[event_details_for_display['Space_Description'] == selected_space]
            else:
                st.selectbox("Filter by Space:", ["No Spaces available"], disabled=True)
        
        # Row 3 of filters: Status and Category Type
        filter_row3_col1, filter_row3_col2 = st.columns(2)
        with filter_row3_col1:
            if not event_details_base.empty and 'Status_Description' in event_details_base.columns:
                unique_statuses = ["All Statuses"] + sorted(event_details_base['Status_Description'].dropna().unique())
                selected_status = st.selectbox(
                    "Filter by Status:",
                    options=unique_statuses,
                    index=0,
                    key="status_filter_selectbox_tab3"
                )
                if selected_status != "All Statuses":
                    event_details_for_display = event_details_for_display[event_details_for_display['Status_Description'] == selected_status]
            else:
                st.selectbox("Filter by Status:", ["No Statuses available"], disabled=True)

        with filter_row3_col2:
            if not event_details_base.empty and 'Category_Type' in event_details_base.columns:
                unique_cat_types = ["All Category Types"] + sorted(event_details_base['Category_Type'].dropna().unique())
                selected_cat_type = st.selectbox(
                    "Filter by Category Type:",
                    options=unique_cat_types,
                    index=0,
                    key="cat_type_filter_selectbox_tab3"
                )
                if selected_cat_type != "All Category Types":
                    event_details_for_display = event_details_for_display[event_details_for_display['Category_Type'] == selected_cat_type]
            else:
                st.selectbox("Filter by Category Type:", ["No Category Types available"], disabled=True)
        
        st.markdown("---") # Visual separator before sort controls

        # --- Sorting controls for the displayed table ---
        col_sort1, col_sort2 = st.columns([3, 1])
        
        # Define the full intended column mapping first for sorting options
        # This includes all columns we might want to display and sort by their friendly names
        potential_column_mapping = {
            'StartDate': 'Date',
            'FunctionStart': 'Time',
            'EndDate': 'End Date',
            'FunctionEnd': 'End Time',
            'EventID': 'Event ID',
            'EventType_Description': 'Event Type',
            'Event_Description': 'Event Name',
            'Function_Description': 'Function Details',
            'OrderedAttendance': 'Expected Attendance',
            'PlannerName_Masked': 'Planner',
            'BusinessGroup_Description': 'Business Group',
            'Space_Description': 'Space',
            'Status_Description': 'Status',
            'Category_Type': 'Category Type',
            'Allocation': 'Allocation',
            'ActualRevenue': 'Revenue',
            'OrderedRevenue': 'Ordered Revenue',
        }

        with col_sort1:
            # Determine available sort options based on columns that will actually be in event_display_df
            # and have a mapping
            # We check against event_details_for_display.columns (original names) that are in potential_column_mapping.keys()
            # and then use their mapped names.
            
            # Define display columns *before* setting sort options
            # These are the *original* column names we intend to show
            display_columns_intended = [
                'StartDate', 'FunctionStart', 'EndDate', 'FunctionEnd', 
                'EventID', 'Event_Description', 'EventType_Description', 'Function_Description',
                'PlannerName_Masked', 'BusinessGroup_Description', 'Space_Description',
                'Status_Description', 'Category_Type', 'Allocation',
                'OrderedAttendance', 'ActualRevenue', 'OrderedRevenue'
            ]
            # Filter to columns that actually exist in the (filtered) dataframe
            actual_display_columns_original_names = [col for col in display_columns_intended if col in event_details_for_display.columns]
            
            # Get the friendly names for sorting options
            sort_by_friendly_options = [potential_column_mapping[col] for col in actual_display_columns_original_names if col in potential_column_mapping]
            if not sort_by_friendly_options: sort_by_friendly_options = ["Date"] # Default

            sort_by = st.selectbox("Sort by", sort_by_friendly_options, index=0 if "Date" in sort_by_friendly_options else 0, key="sort_by_tab3")
        
        with col_sort2:
            ascending = st.checkbox("Ascending order", value=True, key="ascending_tab3")
        
        # --- Create a clean dataframe for display with selected columns ---
        if not event_details_for_display.empty:
            # Use actual_display_columns_original_names which are guaranteed to exist
            event_display_df = event_details_for_display[actual_display_columns_original_names].copy()
            
            # Rename columns for better readability using the potential_column_mapping
            # Only rename columns that are present in event_display_df and have a mapping
            rename_dict_final = {k: v for k, v in potential_column_mapping.items() if k in event_display_df.columns}
            event_display_df.rename(columns=rename_dict_final, inplace=True)
            
            # Format date and time columns
            if 'Date' in event_display_df.columns: # Corresponds to StartDate
                event_display_df['Date'] = pd.to_datetime(event_display_df['Date']).dt.strftime('%a, %b %d, %Y')
            if 'End Date' in event_display_df.columns: # Corresponds to EndDate
                event_display_df['End Date'] = pd.to_datetime(event_display_df['End Date']).dt.strftime('%a, %b %d, %Y')
            
            if 'Time' in event_display_df.columns: # Corresponds to FunctionStart
                event_display_df['Time'] = pd.to_datetime(event_display_df['Time']).dt.strftime('%I:%M %p')
            if 'End Time' in event_display_df.columns: # Corresponds to FunctionEnd
                event_display_df['End Time'] = pd.to_datetime(event_display_df['End Time']).dt.strftime('%I:%M %p')

            # Reorder columns to group date/time pairs
            final_columns_ordered = []
            current_cols_set = set(event_display_df.columns)
            
            # Define preferred order, checking existence
            preferred_order = [
                'Date', 'Time', 'End Date', 'End Time', 'Event ID', 'Event Name', 'Event Type', 
                'Function Details', 'Planner', 'Business Group', 'Space', 'Status',
                'Category Type', 'Allocation', 'Expected Attendance', 'Revenue', 'Ordered Revenue'
            ]
            for col_name in preferred_order:
                if col_name in current_cols_set:
                    final_columns_ordered.append(col_name)
                    current_cols_set.remove(col_name)
            final_columns_ordered.extend(sorted(list(current_cols_set))) # Add any remaining columns, sorted
            
            event_display_df = event_display_df[final_columns_ordered]

            # Format revenue columns 
            if 'Revenue' in event_display_df.columns:
                event_display_df['Revenue'] = event_display_df['Revenue'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "$0")
            if 'Ordered Revenue' in event_display_df.columns:
                event_display_df['Ordered Revenue'] = event_display_df['Ordered Revenue'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "$0")
            
            # --- Sort the dataframe based on user selection ---
            # Map friendly sort_by name back to an original name if necessary for sorting logic that uses original values (e.g. actual date objects)
            # However, for most cases, sorting on the formatted event_display_df is fine.
            # For 'Date' or 'End Date', it's better to sort on the original unformatted datetime objects if strict chronological order is key
            # The current implementation sorts 'Date' (StartDate) on event_details_for_display which is good.
            # Let's adjust to handle if the user selects 'End Date' for sorting.

            sort_col_original_for_date_types = None
            if sort_by == "Date": # Mapped from StartDate
                 sort_col_original_for_date_types = 'StartDate'
            elif sort_by == "End Date": # Mapped from EndDate
                 sort_col_original_for_date_types = 'EndDate'

            if sort_col_original_for_date_types and sort_col_original_for_date_types in event_details_for_display.columns:
                # Sort the source dataframe by the original date column
                # then rebuild event_display_df from this sorted source.
                # This ensures correct chronological sorting before formatting.
                sorted_indices = event_details_for_display[sort_col_original_for_date_types].sort_values(ascending=ascending).index
                event_display_df = event_display_df.loc[sorted_indices]
            elif sort_by in event_display_df.columns: # For non-date or already formatted date sort
                # Handle numeric sorting for revenue columns correctly
                if sort_by == "Revenue" or sort_by == "Ordered Revenue":
                    temp_sort_series = pd.to_numeric(event_display_df[sort_by].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False), errors='coerce')
                    event_display_df = event_display_df.iloc[temp_sort_series.sort_values(ascending=ascending).index]
                elif sort_by == "Expected Attendance": # Ensure numeric sort
                     temp_sort_series = pd.to_numeric(event_display_df[sort_by], errors='coerce')
                     event_display_df = event_display_df.iloc[temp_sort_series.sort_values(ascending=ascending).index]
                else: # Default sort for other columns
                    event_display_df = event_display_df.sort_values(by=sort_by, ascending=ascending, na_position='last')
            
            # Display the dataframe
            st.dataframe(event_display_df, use_container_width=True, hide_index=True)

        else:
            st.info("No events match the current filter criteria. Please adjust your filters in the sidebar or the Event Type filter on this tab.")

        # --- Chatbot Section for Event Details ---
        st.markdown("---")
        st.subheader("💬 Chat with Event Details")

        if "event_detail_messages" not in st.session_state:
            st.session_state.event_detail_messages = []

        # Display existing messages
        for message in st.session_state.event_detail_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask about the event data shown above..."):
            # Check for API key
            if not st.session_state.openai_api_key:
                st.error("Please enter your OpenAI API Key in the sidebar to use the chatbot.")
            elif event_details_for_display.empty:
                st.warning("There is no data currently displayed in the table to chat about. Please adjust filters.")
            else:
                # Add user message to chat history
                st.session_state.event_detail_messages.append({"role": "user", "content": prompt})
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.spinner("Preparing data and thinking..."):
                    # Prepare data for LLM
                    # Use the dataframe that is currently displayed in the table (already filtered)
                    data_for_llm = event_display_df.to_csv(index=False)

                    system_prompt_chat = f"""You are an AI assistant specialized in analyzing event data. 
The user has provided a table of event details (in CSV format below) that has already been filtered based on their selections. 
Your task is to answer questions based ONLY on this provided data. 
Be concise and directly address the user's question using the data. 
If the data does not contain the answer, explicitly state that. 
Do not make assumptions or use external knowledge.
Today's date is {datetime.now().strftime('%Y-%m-%d')}. Consider this if asked about past/future/current events relative to today.
Provided Event Data (CSV):
{data_for_llm}
"""
                    messages_for_api = [
                        {"role": "system", "content": system_prompt_chat},
                        # Include recent chat history for context, if any
                        # For simplicity, let's just send the current prompt for now, but history could be added here.
                        {"role": "user", "content": prompt}
                    ]

                    try:
                        client = OpenAI(api_key=st.session_state.openai_api_key)
                        with st.chat_message("assistant"):
                            with st.spinner("Thinking..."):
                                response = client.chat.completions.create(
                                    model=st.session_state.chatbot_model,
                                    messages=messages_for_api,
                                    temperature=0.5, # Adjust for creativity vs. factuality
                                )
                            ai_response = response.choices[0].message.content
                            st.markdown(ai_response)
                            st.session_state.event_detail_messages.append({"role": "assistant", "content": ai_response})
                    except Exception as e:
                        st.error(f"Error communicating with OpenAI: {e}")
                        # Optionally remove the last user message if the API call failed severely
                        # st.session_state.event_detail_messages.pop()

    # --- ATTENDEE DASHBOARD TAB ---
    with tab_attendee_dashboard:
        st.markdown('<div class="tab-page-title">Attendee Engagement Dashboard</div>', unsafe_allow_html=True)

        if not current_filters.empty and 'OrderedAttendance' in df_event_max_attendance.columns and \
           'ActualRevenue' in df_event_max_attendance.columns and 'EventID' in df_event_max_attendance.columns:
            
            # --- Calculate metrics for KPI cards using df_event_max_attendance ---
            total_attendees_val_dax = df_event_max_attendance['OrderedAttendance'].sum()
            total_distinct_events_for_avg = df_event_max_attendance['EventID'].nunique()
            avg_attendees_per_event_val = (total_attendees_val_dax / total_distinct_events_for_avg) if total_distinct_events_for_avg > 0 else 0
            total_actual_revenue_for_kpi = df_event_max_attendance['ActualRevenue'].sum() # Use sum of revenue from the same df for consistency here
            avg_revenue_per_attendee_val = (total_actual_revenue_for_kpi / total_attendees_val_dax) if total_attendees_val_dax > 0 else 0

            # Metric cards remain the same calculation logic, but now based on df_event_max_attendance implicitly
            kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
            with kpi_col1: display_total_attendees = f"{total_attendees_val_dax/1000:.0f}K" if total_attendees_val_dax >= 1000 else str(total_attendees_val_dax); st.markdown(f'''<div class="metric-card"><p>TOTAL ATTENDEES</p><h2>{display_total_attendees}</h2></div>''', unsafe_allow_html=True)
            with kpi_col2: st.markdown(f'''<div class="metric-card"><p>AVG ATTENDEES PER EVENT</p><h2>{avg_attendees_per_event_val:.2f}</h2></div>''', unsafe_allow_html=True)
            with kpi_col3: st.markdown(f'''<div class="metric-card"><p>AVG REVENUE / ATTENDEE</p><h2>${avg_revenue_per_attendee_val:.2f}</h2></div>''', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            st.subheader("Analysis by Event Type")
            # Use df_event_max_attendance for charts
            attendee_chart_data_tab4 = df_event_max_attendance.copy()

            if not attendee_chart_data_tab4.empty:
                # Chart 1: Actual Revenue by EventType (using max attendance df for consistency)
                revenue_by_event_type = attendee_chart_data_tab4.groupby('EventType_Description')['ActualRevenue'].sum().reset_index().sort_values(by='ActualRevenue', ascending=False)
                # Chart 2: Revenue per Attendee (using max attendance)
                attendee_chart_data_tab4_filt = attendee_chart_data_tab4[attendee_chart_data_tab4['OrderedAttendance'] > 0]
                if not attendee_chart_data_tab4_filt.empty:
                    revenue_per_attendee_by_type = attendee_chart_data_tab4_filt.groupby('EventType_Description').agg(TotalRevenue=('ActualRevenue', 'sum'), TotalAttendees=('OrderedAttendance', 'sum')).reset_index()
                    revenue_per_attendee_by_type['RevenuePerAttendee'] = revenue_per_attendee_by_type['TotalRevenue'] / revenue_per_attendee_by_type['TotalAttendees']
                    revenue_per_attendee_by_type = revenue_per_attendee_by_type.sort_values(by='RevenuePerAttendee', ascending=False)
                else: revenue_per_attendee_by_type = pd.DataFrame(columns=['EventType_Description', 'RevenuePerAttendee'])
                # Chart 3: Avg Attendees (using max attendance)
                avg_attendees_by_type = attendee_chart_data_tab4.groupby('EventType_Description')['OrderedAttendance'].mean().reset_index().sort_values(by='OrderedAttendance', ascending=False)
                # Chart 4: Event Count by Attendance Bucket (using max attendance)
                bins = [0, 10, 25, 50, 100, 200, 500, float('inf')]; labels = ['1-10', '11-25', '26-50', '51-100', '101-200', '201-500', '>500']
                attendee_chart_data_tab4['AttendanceBucket'] = pd.cut(attendee_chart_data_tab4['OrderedAttendance'], bins=bins, labels=labels, right=True)
                event_count_by_bucket = attendee_chart_data_tab4.groupby('AttendanceBucket', observed=False)['EventID'].nunique().reset_index(name='EventCount')

                # Layout and Plotting (remains mostly same, just uses data derived from df_event_max_attendance)
                chart_cols1 = st.columns(2); chart_cols2 = st.columns(2)
                with chart_cols1[0]: create_bar_chart(revenue_by_event_type.head(10), y_col='EventType_Description', x_col='ActualRevenue', title='Actual Revenue by Event Type (Top 10)', x_label='Total Revenue ($', text_template='$%{text:,.0s}', color='#6A5ACD')
                with chart_cols1[1]: create_bar_chart(revenue_per_attendee_by_type.head(10), y_col='EventType_Description', x_col='RevenuePerAttendee', title='Revenue per Attendee by Event Type (Top 10)', x_label='Revenue per Attendee ($', text_template='$%{text:.2f}', color='#7B68EE') # Updated format
                with chart_cols2[0]: create_bar_chart(avg_attendees_by_type.head(10), y_col='EventType_Description', x_col='OrderedAttendance', title='Avg Attendees per Event by Type (Top 10)', x_label='Avg Attendees', text_template='%{text:.1f}', color='#8A2BE2') # Updated format
                with chart_cols2[1]: 
                    if not event_count_by_bucket.empty: fig_event_count_bucket = px.bar(event_count_by_bucket, x='AttendanceBucket', y='EventCount', title='Event Count by Attendance Bucket', labels={'EventCount': 'Number of Events', 'AttendanceBucket': 'Attendance Bucket'}); fig_event_count_bucket.update_traces(marker_color='#9370DB'); fig_event_count_bucket.update_layout(height=400, plot_bgcolor='white', margin=dict(l=40, r=20, t=50, b=20)); st.plotly_chart(fig_event_count_bucket, use_container_width=True)
                    else: st.info("No data for 'Event Count by Attendance Bucket'.")
            else: st.info("No data available for event type visualizations based on current filters.")
        elif current_filters.empty: st.info("No data matches the current filter criteria.")
        else: st.warning("Required columns for attendee analysis missing.")

    # --- REVENUE DASHBOARD TAB (tab5) ---
    with tab_revenue_dashboard:
        # This tab uses df_event_level_kpi for its distinct event revenue counts, which is correct for its purpose.
        # No changes needed here based on the DAX logic for attendance.
        st.markdown('<div class="tab-page-title">Revenue Performance Dashboard</div>', unsafe_allow_html=True)

        if not current_filters.empty:
            # Use df_event_level_kpi for metrics that should be based on distinct events
            # df_event_level_kpi is already filtered based on sidebar selections
            
            # --- Calculate metrics for KPI cards ---
            total_actual_revenue_rd = df_event_level_kpi['ActualRevenue'].sum() if 'ActualRevenue' in df_event_level_kpi.columns else 0
            total_ordered_revenue_rd = df_event_level_kpi['OrderedRevenue'].sum() if 'OrderedRevenue' in df_event_level_kpi.columns else 0
            total_forecast_revenue_rd = df_event_level_kpi['ForecastRevenue'].sum() if 'ForecastRevenue' in df_event_level_kpi.columns else 0

            # Metric cards for Revenue Dashboard
            kpi_rd_col1, kpi_rd_col2, kpi_rd_col3 = st.columns(3)
            with kpi_rd_col1: st.markdown(f'''<div class="metric-card"><p>TOTAL ACTUAL REVENUE</p><h2>${total_actual_revenue_rd:,.0f}</h2></div>''', unsafe_allow_html=True) # Changed to ,.0f for consistency
            with kpi_rd_col2: st.markdown(f'''<div class="metric-card"><p>TOTAL ORDERED REVENUE</p><h2>${total_ordered_revenue_rd:,.0f}</h2></div>''', unsafe_allow_html=True)
            with kpi_rd_col3: st.markdown(f'''<div class="metric-card"><p>TOTAL FORECAST REVENUE</p><h2>${total_forecast_revenue_rd:,.0f}</h2></div>''', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            st.subheader("Revenue & Event Count Breakdowns")
            revenue_chart_data_tab5 = df_event_level_kpi.copy() 

            if not revenue_chart_data_tab5.empty:
                # Chart 1: Total Actual Revenue by EventType_Description
                actual_rev_by_event_type_tab5 = revenue_chart_data_tab5.groupby('EventType_Description')['ActualRevenue'].sum().reset_index().sort_values(by='ActualRevenue', ascending=False)
                # Chart 2: Top Business Groups (by Total Actual Revenue)
                actual_rev_by_biz_group_tab5 = revenue_chart_data_tab5.groupby('BusinessGroup_Description')['ActualRevenue'].sum().reset_index().sort_values(by='ActualRevenue', ascending=False)
                # Chart 3: Event Count by BG_Desc_Masked
                event_count_by_bg_masked_tab5 = revenue_chart_data_tab5.groupby('BG_Desc_Masked')['EventID'].nunique().reset_index(name='EventCount').sort_values(by='EventCount', ascending=False)

                chart_rd_row1_col1, chart_rd_row1_col2 = st.columns(2)
                with chart_rd_row1_col1:
                    create_bar_chart(actual_rev_by_event_type_tab5.head(7), y_col='EventType_Description', x_col='ActualRevenue', title='Total Actual Revenue by Event Type', x_label='Total Actual Revenue ($', text_template='$%{text:,.0s}', color='#FF8C00')
                with chart_rd_row1_col2:
                    create_bar_chart(actual_rev_by_biz_group_tab5.head(7), y_col='BusinessGroup_Description', x_col='ActualRevenue', title='Total Actual Revenue by Business Group', x_label='Total Actual Revenue ($', text_template='$%{text:,.0s}', color='#FF7F50')
                
                # Ensure BG_Desc_Masked exists before trying to plot
                if 'BG_Desc_Masked' in revenue_chart_data_tab5.columns:
                     create_bar_chart(event_count_by_bg_masked_tab5.head(7), y_col='BG_Desc_Masked', x_col='EventCount', title='Event Count by BG Desc Masked', x_label='Number of Events', text_template='%{text}', color='#FFA07A', full_width=True)
                else:
                    st.info("'BG_Desc_Masked' column not found for chart.")
                
                st.markdown("--- ")
                # Chart 4: Line chart for Revenue Trends
                line_chart_data_tab5 = revenue_chart_data_tab5.copy()
                if 'StartDate' in line_chart_data_tab5.columns:
                    line_chart_data_tab5['MonthYear'] = pd.to_datetime(line_chart_data_tab5['StartDate']).dt.to_period('M').dt.to_timestamp()
                    revenue_over_time_tab5 = line_chart_data_tab5.groupby('MonthYear').agg(ActualRevenue=('ActualRevenue', 'sum'), OrderedRevenue=('OrderedRevenue', 'sum'), ForecastRevenue=('ForecastRevenue', 'sum')).reset_index()
                    create_line_chart(revenue_over_time_tab5, title='Revenue Trends by Month (Actual, Ordered, Forecast)') # Added Forecast to title
                else:
                    st.info("'StartDate' column required for Revenue Trends chart is missing.")

                st.markdown("--- ")
                # Chart 5: Actual Revenue by Category_Type
                if 'Category_Type' in revenue_chart_data_tab5.columns:
                    actual_rev_by_cat_type_tab5 = revenue_chart_data_tab5.groupby('Category_Type')['ActualRevenue'].sum().reset_index().sort_values(by='ActualRevenue', ascending=False)
                    create_bar_chart(actual_rev_by_cat_type_tab5.head(7), y_col='Category_Type', x_col='ActualRevenue', title='Total Actual Revenue by Category Type', x_label='Total Actual Revenue ($', text_template='$%{text:,.0s}', color='#CD5C5C', full_width=True)
                else:
                    st.info("'Category_Type' column not found for chart.")
            else:
                st.info("No data for revenue breakdown visualizations based on current filters.")
        else:
            st.info("No data matches the current filter criteria to display Revenue Performance Dashboard.")

    # --- ADVANCED ANALYTICS TAB (tab6) ---
    with tab_advanced_analytics:
        st.markdown('<div class="tab-page-title">Advanced Analytics Dashboard</div>', unsafe_allow_html=True)

        if not current_filters.empty:
            # --- Cumulative Revenue by Event Type (Waterfall Chart) - ENSURE IT IS AT THE TOP ---
            st.subheader("Cumulative Revenue Contribution by Event Type")
            if 'EventType_Description' in df_event_level_kpi.columns and 'ActualRevenue' in df_event_level_kpi.columns:
                revenue_by_event_type_wc = df_event_level_kpi.groupby('EventType_Description')['ActualRevenue'].sum().sort_values(ascending=False)
                if not revenue_by_event_type_wc.empty:
                    top_n_waterfall = 7
                    wc_data = revenue_by_event_type_wc.head(top_n_waterfall).copy()
                    if len(revenue_by_event_type_wc) > top_n_waterfall:
                        other_sum = revenue_by_event_type_wc.iloc[top_n_waterfall:].sum()
                        if other_sum > 0: wc_data['Other'] = other_sum
                    
                    if 'Other' in wc_data and wc_data['Other'] == 0 and len(wc_data) > 1: # Avoid dropping if 'Other' is the only item
                         wc_data.drop('Other', inplace=True)

                    if not wc_data.empty: # Proceed only if wc_data is not empty after potential drop
                        fig_waterfall = go.Figure(go.Waterfall(
                            name="Revenue", orientation="h", 
                            measure=["relative"] * len(wc_data) + ["total"], 
                            y=wc_data.index.tolist() + ["Total Revenue"], 
                            x=wc_data.values.tolist() + [wc_data.sum()],
                            connector={"line":{"color":"rgb(63, 63, 63)"}},
                            increasing={"marker":{"color":"#FF8C00"}}, 
                            totals={"marker":{"color":"#4A235A"}}
                        ))
                        fig_waterfall.update_layout(title="Cumulative Sum of Total Revenue by Event Type", yaxis_title="Event Type", xaxis_title="Sum of Total Revenue ($)", height=500, plot_bgcolor='white', yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig_waterfall, use_container_width=True)
                    else:
                        st.info("No significant data for Cumulative Revenue Waterfall chart after processing.")        
                else: st.info("No data for Cumulative Revenue Waterfall chart.")
            else: st.warning("'EventType_Description' or 'ActualRevenue' column not available for Waterfall chart.")
            
            st.markdown("--- ") # Visual separator

            # --- Heatmaps for Revenue and Avg Attendees by DayOfWeek/Month (Displayed AFTER Waterfall) ---
            st.subheader("Revenue & Attendance Patterns by Day of Week and Month")
            # Use df_event_max_attendance for Avg Attendees heatmap
            if 'StartDate' in df_event_max_attendance.columns:
                heatmap_data_adv = df_event_max_attendance.copy() # Use max attendance data
                heatmap_data_adv['DayOfWeekName'] = pd.to_datetime(heatmap_data_adv['StartDate']).dt.day_name()
                days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                heatmap_data_adv['DayOfWeekName'] = pd.Categorical(heatmap_data_adv['DayOfWeekName'], categories=days_order, ordered=True)
                heatmap_data_adv['MonthNum'] = pd.to_datetime(heatmap_data_adv['StartDate']).dt.month
                heatmap_col1, heatmap_col2 = st.columns(2)

                with heatmap_col1:
                    # Revenue heatmap: Use df_event_level_kpi as it's revenue-focused
                    if 'ActualRevenue' in df_event_level_kpi.columns and 'StartDate' in df_event_level_kpi.columns:
                        # Prepare data for this specific heatmap
                        revenue_heatmap_df = df_event_level_kpi.copy()
                        revenue_heatmap_df['DayOfWeekName'] = pd.to_datetime(revenue_heatmap_df['StartDate']).dt.day_name()
                        revenue_heatmap_df['MonthNum'] = pd.to_datetime(revenue_heatmap_df['StartDate']).dt.month
                        
                        revenue_heatmap_pivot_adv = pd.pivot_table(
                            revenue_heatmap_df, 
                            values='ActualRevenue', 
                            index='DayOfWeekName', # Use the column directly
                            columns='MonthNum', 
                            aggfunc='sum', 
                            fill_value=0
                        ).reindex(index=days_order, fill_value=0) # Reindex AFTER pivoting
                        
                        if not revenue_heatmap_pivot_adv.empty:
                            fig_rev_heatmap_adv = px.imshow(revenue_heatmap_pivot_adv, title="Revenue by Day & Month", labels=dict(x="Month", y="Day of Week", color="Total Revenue"), color_continuous_scale="Oranges", text_auto='.2s')
                            fig_rev_heatmap_adv.update_layout(height=450, plot_bgcolor='white', margin=dict(t=50, b=10, l=10, r=10))
                            fig_rev_heatmap_adv.update_xaxes(side="top", tickmode='array', tickvals=list(range(1,13)), ticktext=[datetime(2000,i,1).strftime('%b') for i in range(1,13)])
                            st.plotly_chart(fig_rev_heatmap_adv, use_container_width=True)
                        else: st.info("No data for Revenue heatmap.")
                    else: st.warning("'ActualRevenue' or 'StartDate' missing for Revenue heatmap.")
                
                with heatmap_col2:
                    # Attendee heatmap uses df_event_max_attendance
                    if 'OrderedAttendance' in heatmap_data_adv.columns and 'StartDate' in heatmap_data_adv.columns: # heatmap_data_adv IS df_event_max_attendance here
                        avg_attendees_heatmap_pivot_adv = pd.pivot_table(
                            heatmap_data_adv, 
                            values='OrderedAttendance', 
                            index='DayOfWeekName', # Use the column directly
                            columns='MonthNum', 
                            aggfunc='mean', 
                            fill_value=0 
                        ).reindex(index=days_order, fill_value=0) # Reindex AFTER pivoting
                        
                        if not avg_attendees_heatmap_pivot_adv.empty:
                            fig_att_heatmap_adv = px.imshow(avg_attendees_heatmap_pivot_adv, title="Avg Attendees by Day & Month", labels=dict(x="Month", y="Day of Week", color="Avg Attendees"), color_continuous_scale="Purples", text_auto='.1f')
                            fig_att_heatmap_adv.update_layout(height=450, plot_bgcolor='white', margin=dict(t=50, b=10, l=10, r=10))
                            fig_att_heatmap_adv.update_xaxes(side="top", tickmode='array', tickvals=list(range(1,13)), ticktext=[datetime(2000,i,1).strftime('%b') for i in range(1,13)])
                            st.plotly_chart(fig_att_heatmap_adv, use_container_width=True)
                        else: st.info("No data for Attendee heatmap.")
                    else: st.warning("'OrderedAttendance' or 'StartDate' missing for Attendee heatmap.")
            else:
                st.warning("'StartDate' column not available for heatmap generation.")
            
            st.markdown("--- ")
            # --- Outlier Detection for Daily Ordered Attendance ---
            # This uses current_filters (raw filtered data) - Summing daily total attendance is correct here.
            # No change needed.
            st.subheader("Outlier Detection in Daily Ordered Attendance")
            if 'OrderedAttendance' in current_filters.columns and 'StartDate' in current_filters.columns:
                daily_att_df = current_filters.copy()
                daily_att_df['Date'] = pd.to_datetime(daily_att_df['StartDate']).dt.date
                daily_attendance = daily_att_df.groupby('Date')['OrderedAttendance'].sum().reset_index()
                daily_attendance.rename(columns={'OrderedAttendance': 'TotalOrderedAttendance'}, inplace=True)
                daily_attendance = daily_attendance.set_index('Date')
                if not daily_attendance.empty:
                    mean_att = daily_attendance['TotalOrderedAttendance'].mean()
                    std_att = daily_attendance['TotalOrderedAttendance'].std()
                    outlier_threshold = mean_att + (3 * std_att)
                    daily_attendance['IsOutlier'] = daily_attendance['TotalOrderedAttendance'] > outlier_threshold
                    outliers_df = daily_attendance[daily_attendance['IsOutlier']]
                    fig_outlier = go.Figure()
                    fig_outlier.add_trace(go.Scatter(x=daily_attendance.index, y=daily_attendance['TotalOrderedAttendance'], mode='lines', name='Daily Attendance', line=dict(color='teal')))
                    if not outliers_df.empty: fig_outlier.add_trace(go.Scatter(x=outliers_df.index, y=outliers_df['TotalOrderedAttendance'], mode='markers', name='Outlier (> mean + 3σ)', marker=dict(color='red', size=8)))
                    fig_outlier.add_hline(y=outlier_threshold, line_dash="dash", line_color="grey", annotation_text=f"Threshold (μ+3σ): {outlier_threshold:,.0f}", annotation_position="bottom right")
                    fig_outlier.update_layout(title='Daily Ordered Attendance (Outliers > mean + 3σ)', xaxis_title='Date', yaxis_title='Total Ordered Attendance', height=500, plot_bgcolor='white', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
                    st.plotly_chart(fig_outlier, use_container_width=True)
                else: st.info("Not enough daily attendance data for outlier detection.")
            else: st.warning("'OrderedAttendance' or 'StartDate' missing for outlier detection.")
            
            st.markdown("--- ")
            # --- Business Group Impact Analysis ---
            # Update this section to use df_event_max_attendance for attendance calculations
            st.subheader("Business Group Impact Analysis")
            if not df_event_max_attendance.empty and 'BusinessGroup_Description' in df_event_max_attendance.columns and 'OrderedAttendance' in df_event_max_attendance.columns and 'ActualRevenue' in df_event_max_attendance.columns and 'EventID' in df_event_max_attendance.columns:
                # Aggregate using df_event_max_attendance
                grouped_by_biz_adv = df_event_max_attendance.groupby('BusinessGroup_Description').agg(
                    Sum_ActualRevenue = ('ActualRevenue', 'sum'),
                    Sum_OrderedAttendance = ('OrderedAttendance', 'sum'), # Now sums the MAX attendance per event for the group
                    TotalEvents = ('EventID', 'nunique')
                ).reset_index()
                grouped_by_biz_adv['AvgAttendeesPerEvent'] = (grouped_by_biz_adv['Sum_OrderedAttendance'] / grouped_by_biz_adv['TotalEvents']).fillna(0)
                grouped_by_biz_adv['ImpactScore'] = (grouped_by_biz_adv['Sum_ActualRevenue'] / grouped_by_biz_adv['TotalEvents']).fillna(0)
                grouped_by_biz_adv['RevenuePerAttendee'] = (grouped_by_biz_adv['Sum_ActualRevenue'] / grouped_by_biz_adv['Sum_OrderedAttendance']).replace([np.inf, -np.inf], 0).fillna(0)
                
                if not grouped_by_biz_adv.empty:
                    # Plotting uses the newly calculated grouped_by_biz_adv
                    fig_impact_scatter = px.scatter(grouped_by_biz_adv, x='AvgAttendeesPerEvent', y='Sum_ActualRevenue', size='ImpactScore', color='BusinessGroup_Description', hover_name='BusinessGroup_Description', hover_data={'BusinessGroup_Description': False, 'AvgAttendeesPerEvent': ':.2f', 'Sum_ActualRevenue': ':,.0f', 'TotalEvents': True, 'ImpactScore': ':.2f', 'RevenuePerAttendee': ':.2f'}, title='Business Group Performance: Revenue, Attendance & Impact', labels={'AvgAttendeesPerEvent': 'Avg Attendees per Event', 'Sum_ActualRevenue': 'Total Actual Revenue ($', 'ImpactScore': 'Impact Score (Revenue/Event)'}, size_max=60)
                    fig_impact_scatter.update_layout(height=600, plot_bgcolor='white', legend_title_text='Business Group')
                    st.plotly_chart(fig_impact_scatter, use_container_width=True)
                else: st.info("No data for Business Group Impact Analysis chart.")
            else: st.warning("Required columns/data missing for Business Group Impact Analysis.")

            st.markdown("--- ") 
            # --- Cumulative Revenue by Event Type (Waterfall Chart) ---
            # This uses df_event_level_kpi (first occurrence for revenue), which is appropriate for revenue-focused view.
            # No change needed here.

        else:
            st.info("No data matches the current filter criteria. Please adjust your filters in the sidebar.")

        # --- Chatbot Section for Advanced Analytics Tab ---
        st.markdown("---")
        st.subheader("💬 Chat with Advanced Analytics Data")

        if "advanced_analytics_messages" not in st.session_state:
            st.session_state.advanced_analytics_messages = []

        # Display existing messages
        for message in st.session_state.advanced_analytics_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input for advanced analytics
        if prompt_advanced := st.chat_input("Ask about the advanced analytics shown above..."):
            if not st.session_state.openai_api_key:
                st.error("Please enter your OpenAI API Key in the sidebar to use the chatbot.")
            # Check if the necessary dataframes for this tab's charts have been computed
            # We'll use the presence of the figures themselves or key dataframes in session state if we adapt to store them
            elif ('daily_attendance' not in locals() and 'daily_attendance' not in globals()) and \
                 ('grouped_by_biz_adv' not in locals() and 'grouped_by_biz_adv' not in globals()):
                st.warning("Please ensure data is loaded and filters are set to generate the advanced analytics charts before using the chatbot.")
            else:
                st.session_state.advanced_analytics_messages.append({"role": "user", "content": prompt_advanced})
                with st.chat_message("user"):
                    st.markdown(prompt_advanced)

                with st.spinner("Preparing data and thinking..."):
                    outlier_data_csv = ""
                    impact_data_csv = ""
                    
                    # Prepare Outlier Data if available
                    # Check if 'daily_attendance' was computed and is not empty
                    daily_attendance_df_for_llm = None
                    if 'daily_attendance' in locals() or 'daily_attendance' in globals():
                        # Ensure 'daily_attendance' is the DataFrame, not the st.line_chart object if named similarly
                        if isinstance(daily_attendance, pd.DataFrame) and not daily_attendance.empty:
                             daily_attendance_df_for_llm = daily_attendance.reset_index()[['Date', 'TotalOrderedAttendance', 'IsOutlier']].copy()
                             outlier_data_csv = daily_attendance_df_for_llm.to_csv(index=False)


                    # Prepare Impact Analysis Data if available
                    grouped_by_biz_adv_df_for_llm = None
                    if 'grouped_by_biz_adv' in locals() or 'grouped_by_biz_adv' in globals():
                        if isinstance(grouped_by_biz_adv, pd.DataFrame) and not grouped_by_biz_adv.empty:
                            grouped_by_biz_adv_df_for_llm = grouped_by_biz_adv.copy()
                            impact_data_csv = grouped_by_biz_adv_df_for_llm.to_csv(index=False)

                    if not outlier_data_csv and not impact_data_csv:
                        st.warning("No data available from the Advanced Analytics charts to chat about. Please ensure the charts are visible.")
                        # Remove the last user message as we can't proceed
                        if st.session_state.advanced_analytics_messages and st.session_state.advanced_analytics_messages[-1]["role"] == "user":
                            st.session_state.advanced_analytics_messages.pop()
                    else:
                        system_prompt_advanced_chat = f'''You are an AI assistant specialized in analyzing data from an Advanced Analytics dashboard.
The user has access to two main visualizations on this tab:
1.  **Daily Ordered Attendance Outlier Detection**: Shows daily total attendance and highlights outliers (days with attendance > mean + 3 standard deviations).
2.  **Business Group Impact Analysis**: A scatter plot showing Business Groups based on Average Attendees per Event (x-axis), Total Actual Revenue (y-axis), with bubble size representing Impact Score (Revenue per Event), and color representing the Business Group. Hover data includes Revenue per Attendee and Total Events.

You have been provided with the underlying data for these charts in CSV format below.
Your task is to answer questions based ONLY on this provided data.
Be concise and directly address the user's question using the data.
If the data for a specific chart is not available or doesn't contain the answer, explicitly state that.
Do not make assumptions or use external knowledge.
Today's date is {datetime.now().strftime('%Y-%m-%d')}.

Provided Daily Attendance Outlier Data (CSV):
{outlier_data_csv if outlier_data_csv else "Data not available for this chart."}

Provided Business Group Impact Analysis Data (CSV):
{impact_data_csv if impact_data_csv else "Data not available for this chart."}
'''
                        messages_for_api = [
                            {"role": "system", "content": system_prompt_advanced_chat},
                            {"role": "user", "content": prompt_advanced}
                        ]

                        try:
                            client = OpenAI(api_key=st.session_state.openai_api_key)
                            with st.chat_message("assistant"):
                                placeholder = st.empty() # Create a placeholder
                                response_content_stream = ""
                                for chunk in client.chat.completions.create(
                                    model=st.session_state.chatbot_model,
                                    messages=messages_for_api,
                                    temperature=0.3, # More factual for analytics
                                    stream=True,
                                ):
                                    if chunk.choices[0].delta.content is not None:
                                        response_content_stream += chunk.choices[0].delta.content
                                        placeholder.markdown(response_content_stream + "▌")
                                placeholder.markdown(response_content_stream)
                            st.session_state.advanced_analytics_messages.append({"role": "assistant", "content": response_content_stream})
                        except Exception as e:
                            st.error(f"Error communicating with OpenAI: {e}")
                            if st.session_state.advanced_analytics_messages and st.session_state.advanced_analytics_messages[-1]["role"] == "user":
                                 st.session_state.advanced_analytics_messages.pop()

    # --- WEATHER IMPACT ANALYSIS TAB ---
    with tab_weather_impact:
        st.markdown('<div class="tab-page-title">Weather Impact Analysis</div>', unsafe_allow_html=True)

        if df_merged_with_weather.empty or 'weather_condition' not in df_merged_with_weather.columns:
            st.warning("No merged weather and event data available. Please ensure data is loaded and filters are applied, and weather data could be fetched for the data's date range.")
        else:
            analysis_df = df_merged_with_weather.copy()
            event_level_for_impact = analysis_df.drop_duplicates(subset='EventID', keep='first').copy()

            # Convert temperatures from Celsius to Fahrenheit
            def celsius_to_fahrenheit(celsius):
                if pd.isna(celsius):
                    return celsius
                return (celsius * 9/5) + 32

            # Apply temperature conversion to relevant columns
            temp_columns = ['temperature_2m_max', 'temperature_2m_min']
            for col in temp_columns:
                if col in analysis_df.columns:
                    analysis_df[f'{col}_f'] = analysis_df[col].apply(celsius_to_fahrenheit)
                if col in event_level_for_impact.columns:
                    event_level_for_impact[f'{col}_f'] = event_level_for_impact[col].apply(celsius_to_fahrenheit)

            # Add new weather KPIs section
            st.markdown("### Weather Metrics Overview")
            
            # Calculate average weather metrics (now in Fahrenheit)
            avg_max_temp_f = analysis_df['temperature_2m_max_f'].mean() if 'temperature_2m_max_f' in analysis_df.columns else 0
            avg_min_temp_f = analysis_df['temperature_2m_min_f'].mean() if 'temperature_2m_min_f' in analysis_df.columns else 0
            avg_precip = analysis_df['precipitation_sum'].mean() if 'precipitation_sum' in analysis_df.columns else 0
            avg_max_wind = analysis_df['windspeed_10m_max'].mean() if 'windspeed_10m_max' in analysis_df.columns else 0
            avg_daylight = analysis_df['daylight_time'].mean() if 'daylight_time' in analysis_df.columns else 0
            
            # Display weather KPIs in columns using the same styling as other tabs
            weather_kpi_col1, weather_kpi_col2, weather_kpi_col3, weather_kpi_col4, weather_kpi_col5 = st.columns(5)
            with weather_kpi_col1:
                st.markdown(f'''<div class="metric-card"><p>AVG MAX TEMPERATURE</p><h2>{avg_max_temp_f:.1f}°F</h2></div>''', unsafe_allow_html=True)
            with weather_kpi_col2:
                st.markdown(f'''<div class="metric-card"><p>AVG MIN TEMPERATURE</p><h2>{avg_min_temp_f:.1f}°F</h2></div>''', unsafe_allow_html=True)
            with weather_kpi_col3:
                st.markdown(f'''<div class="metric-card"><p>AVG DAILY PRECIPITATION</p><h2>{avg_precip:.2f}mm</h2></div>''', unsafe_allow_html=True)
            with weather_kpi_col4:
                st.markdown(f'''<div class="metric-card"><p>AVG MAX WIND SPEED</p><h2>{avg_max_wind:.1f}km/h</h2></div>''', unsafe_allow_html=True)
            with weather_kpi_col5:
                st.markdown(f'''<div class="metric-card"><p>AVG DAYLIGHT HOURS</p><h2>{avg_daylight:.1f}h</h2></div>''', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)

            # Event & Revenue Analysis by Detailed Weather Conditions - Added at top
            st.markdown("### Event & Revenue Analysis by Detailed Weather Conditions")
            viz_col1_top, viz_col2_top = st.columns(2)
            with viz_col1_top:
                if 'EventID' in event_level_for_impact.columns and 'weather_condition' in event_level_for_impact.columns:
                    event_count_by_cond = event_level_for_impact.groupby('weather_condition')['EventID'].nunique().reset_index(name='EventCount').sort_values(by='EventCount', ascending=False)
                    fig_event_count_weather = px.bar(event_count_by_cond.head(10), y='weather_condition', x='EventCount', orientation='h', title='Event Count by Weather (Top 10)', labels={'EventCount': 'Number of Events', 'weather_condition': 'Weather'}, text='EventCount')
                    fig_event_count_weather.update_traces(marker_color='#6495ED', texttemplate='%{text}', textposition='outside')
                    fig_event_count_weather.update_layout(height=400, yaxis_title=None, plot_bgcolor='white', margin=dict(l=10, r=10, t=30, b=20), yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_event_count_weather, use_container_width=True, key="weather_impact_event_count_chart_at_top")
                else: st.info("Data missing for 'Event Count by Weather' chart.")
            with viz_col2_top:
                if 'ActualRevenue' in event_level_for_impact.columns and 'weather_condition' in event_level_for_impact.columns:
                    avg_revenue_by_cond = event_level_for_impact.groupby('weather_condition')['ActualRevenue'].mean().reset_index().sort_values(by='ActualRevenue', ascending=False)
                    fig_avg_rev_weather = px.bar(avg_revenue_by_cond.head(10), y='weather_condition', x='ActualRevenue', orientation='h', title='Avg. Event Revenue by Weather (Top 10)', labels={'ActualRevenue': 'Avg. Revenue ($)', 'weather_condition': 'Weather'}, text='ActualRevenue')
                    fig_avg_rev_weather.update_traces(marker_color='#FF7F50', texttemplate='$%{text:,.0f}', textposition='outside')
                    fig_avg_rev_weather.update_layout(height=400, yaxis_title=None, plot_bgcolor='white', margin=dict(l=10, r=10, t=30, b=20), yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_avg_rev_weather, use_container_width=True, key="weather_impact_avg_rev_chart_at_top")
                else: st.info("Data missing for 'Avg. Event Revenue by Weather' chart.")

            st.markdown("<hr style='margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True)

            # NEW: Time Series Analysis Section
            st.markdown("### Time Series Analysis: Weather-Business Ratios & Trends")
            
            # Prepare daily aggregated data for time series
            if 'StartDate' in analysis_df.columns:
                analysis_df['event_date'] = pd.to_datetime(analysis_df['StartDate']).dt.date
                
                # Aggregate daily metrics
                daily_metrics = analysis_df.groupby('event_date').agg({
                    'ActualRevenue': 'sum',
                    'EventID': 'nunique',
                    'OrderedAttendance': 'sum',
                    'temperature_2m_max_f': 'first',  # Using Fahrenheit
                    'temperature_2m_min_f': 'first',  # Using Fahrenheit
                    'precipitation_sum': 'first',
                    'windspeed_10m_max': 'first'
                }).reset_index()
                
                daily_metrics.rename(columns={
                    'ActualRevenue': 'Daily_Revenue',
                    'EventID': 'Daily_Events',
                    'OrderedAttendance': 'Daily_Attendance'
                }, inplace=True)
                
                # Calculate ratio metrics (avoid division by zero)
                daily_metrics['Revenue_per_Temp'] = daily_metrics['Daily_Revenue'] / daily_metrics['temperature_2m_max_f'].replace(0, np.nan)
                daily_metrics['Events_per_Temp'] = daily_metrics['Daily_Events'] / daily_metrics['temperature_2m_max_f'].replace(0, np.nan)
                daily_metrics['Revenue_per_Wind'] = daily_metrics['Daily_Revenue'] / daily_metrics['windspeed_10m_max'].replace(0, np.nan)
                daily_metrics['Events_per_Wind'] = daily_metrics['Daily_Events'] / daily_metrics['windspeed_10m_max'].replace(0, np.nan)
                
                # Calculate weather efficiency metrics
                daily_metrics['Temp_Revenue_Efficiency'] = daily_metrics['Daily_Revenue'] / (daily_metrics['temperature_2m_max_f'] + 1)  # +1 to avoid zero
                daily_metrics['Precip_Impact_on_Events'] = daily_metrics['Daily_Events'] / (daily_metrics['precipitation_sum'] + 1)  # +1 to avoid zero
                
                if not daily_metrics.empty:
                    # Create time series plots
                    ts_col1, ts_col2 = st.columns(2)
                    
                    with ts_col1:
                        # Revenue per Temperature over time
                        fig_rev_temp_ratio = px.line(daily_metrics, x='event_date', y='Revenue_per_Temp',
                                                   title='Revenue Efficiency per Temperature ($/°F) Over Time',
                                                   labels={'Revenue_per_Temp': 'Revenue per °F ($)', 'event_date': 'Date'},
                                                   hover_data={
                                                       'Daily_Revenue': ':$,.0f',
                                                       'temperature_2m_max_f': ':.1f°F',
                                                       'Revenue_per_Temp': ':$,.2f'
                                                   })
                        fig_rev_temp_ratio.update_traces(line=dict(color='orange', width=2))
                        fig_rev_temp_ratio.update_layout(height=400, plot_bgcolor='white')
                        fig_rev_temp_ratio.update_traces(hovertemplate='<b>Date:</b> %{x}<br>' +
                                                        '<b>Revenue/Temp Ratio:</b> $%{y:,.2f}<br>' +
                                                        '<b>Daily Revenue:</b> %{customdata[0]:$,.0f}<br>' +
                                                        '<b>Max Temperature:</b> %{customdata[1]:.1f}°F<extra></extra>',
                                                        customdata=daily_metrics[['Daily_Revenue', 'temperature_2m_max_f']].values)
                        st.plotly_chart(fig_rev_temp_ratio, use_container_width=True)
                    
                    
                    with ts_col2:
                        # Events per Temperature over time
                        fig_events_temp_ratio = px.line(daily_metrics, x='event_date', y='Events_per_Temp',
                                                      title='Event Efficiency per Temperature (Events/°F) Over Time',
                                                      labels={'Events_per_Temp': 'Events per °F', 'event_date': 'Date'})
                        fig_events_temp_ratio.update_traces(line=dict(color='blue', width=2))
                        fig_events_temp_ratio.update_layout(height=400, plot_bgcolor='white')
                        st.plotly_chart(fig_events_temp_ratio, use_container_width=True)
                    
                    # Second row of ratio charts
                    ts_col3, ts_col4 = st.columns(2)
                    
                    with ts_col3:
                        # Revenue performance vs precipitation (inverse relationship)
                        fig_precip_impact = px.line(daily_metrics, x='event_date', y='Precip_Impact_on_Events',
                                                  title='Event Resilience to Precipitation Over Time',
                                                  labels={'Precip_Impact_on_Events': 'Events/(Precipitation+1)', 'event_date': 'Date'})
                        fig_precip_impact.update_traces(line=dict(color='green', width=2))
                        fig_precip_impact.update_layout(height=350, plot_bgcolor='white')
                        st.plotly_chart(fig_precip_impact, use_container_width=True)
                    
                    with ts_col4:
                        # Temperature-adjusted revenue efficiency
                        fig_temp_efficiency = px.line(daily_metrics, x='event_date', y='Temp_Revenue_Efficiency',
                                                    title='Temperature-Adjusted Revenue Efficiency Over Time',
                                                    labels={'Temp_Revenue_Efficiency': 'Revenue/(Temp+1)', 'event_date': 'Date'})
                        fig_temp_efficiency.update_traces(line=dict(color='red', width=2))
                        fig_temp_efficiency.update_layout(height=350, plot_bgcolor='white')
                        st.plotly_chart(fig_temp_efficiency, use_container_width=True)
                else:
                    st.info("Not enough data for time series analysis.")
            
            st.markdown("<hr style='margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True)

            # NEW: Weather-Scaled Business Metrics Heatmaps
            st.markdown("### Business Metrics by Day & Month (Weather-Scaled)")
            st.info("💡 These heatmaps show revenue/attendance values, but the color intensity represents weather conditions")
            
            if 'StartDate' in analysis_df.columns:
                # Prepare combined data for weather-scaled heatmaps
                combined_heatmap_data = analysis_df.copy()
                combined_heatmap_data['DayOfWeekName'] = pd.to_datetime(combined_heatmap_data['StartDate']).dt.day_name()
                days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                combined_heatmap_data['DayOfWeekName'] = pd.Categorical(combined_heatmap_data['DayOfWeekName'], categories=days_order, ordered=True)
                combined_heatmap_data['MonthNum'] = pd.to_datetime(combined_heatmap_data['StartDate']).dt.month
                
                # Create pivot tables for both metrics and weather conditions
                st.subheader("Business Metrics with Weather Color Scaling")
                
                # First, calculate all the pivot tables we'll need
                # Revenue values
                if 'ActualRevenue' in df_event_level_kpi.columns and 'temperature_2m_max_f' in combined_heatmap_data.columns:
                    revenue_heatmap_df = df_event_level_kpi.copy()
                    revenue_heatmap_df['DayOfWeekName'] = pd.to_datetime(revenue_heatmap_df['StartDate']).dt.day_name()
                    revenue_heatmap_df['DayOfWeekName'] = pd.Categorical(revenue_heatmap_df['DayOfWeekName'], categories=days_order, ordered=True)
                    revenue_heatmap_df['MonthNum'] = pd.to_datetime(revenue_heatmap_df['StartDate']).dt.month
                    
                    revenue_pivot = pd.pivot_table(
                        revenue_heatmap_df,
                        values='ActualRevenue',
                        index='DayOfWeekName',
                        columns='MonthNum',
                        aggfunc='sum',
                        fill_value=0
                    ).reindex(index=days_order, fill_value=0)
                    
                    # Temperature values for color scale
                    temp_pivot = pd.pivot_table(
                        combined_heatmap_data,
                        values='temperature_2m_max_f',
                        index='DayOfWeekName',
                        columns='MonthNum',
                        aggfunc='mean',
                        fill_value=np.nan
                    ).reindex(index=days_order, fill_value=np.nan)
                    
                    # Precipitation values for color scale
                    precip_pivot = pd.pivot_table(
                        combined_heatmap_data,
                        values='precipitation_sum',
                        index='DayOfWeekName',
                        columns='MonthNum',
                        aggfunc='mean',
                        fill_value=0
                    ).reindex(index=days_order, fill_value=0)
                    
                    # Wind values for color scale
                    wind_pivot = pd.pivot_table(
                        combined_heatmap_data,
                        values='windspeed_10m_max',
                        index='DayOfWeekName',
                        columns='MonthNum',
                        aggfunc='mean',
                        fill_value=0
                    ).reindex(index=days_order, fill_value=0)
                
                # Attendance values
                attendees_pivot = None
                if 'OrderedAttendance' in df_event_max_attendance.columns:
                    attendees_heatmap_df = df_event_max_attendance.copy()
                    attendees_heatmap_df['DayOfWeekName'] = pd.to_datetime(attendees_heatmap_df['StartDate']).dt.day_name()
                    attendees_heatmap_df['DayOfWeekName'] = pd.Categorical(attendees_heatmap_df['DayOfWeekName'], categories=days_order, ordered=True)
                    attendees_heatmap_df['MonthNum'] = pd.to_datetime(attendees_heatmap_df['StartDate']).dt.month
                    
                    attendees_pivot = pd.pivot_table(
                        attendees_heatmap_df,
                        values='OrderedAttendance',
                        index='DayOfWeekName',
                        columns='MonthNum',
                        aggfunc='mean',
                        fill_value=0
                    ).reindex(index=days_order, fill_value=0)
                
                # Row 1: Temperature-based heatmaps
                st.markdown("##### Temperature Impact")
                temp_row_col1, temp_row_col2 = st.columns(2)
                
                with temp_row_col1:
                    # Revenue with Temperature Color Scale
                    if not revenue_pivot.empty and not temp_pivot.empty:
                        fig_rev_temp_heatmap = go.Figure(data=go.Heatmap(
                            z=temp_pivot.values,  # Color based on temperature
                            x=[datetime(2000,i,1).strftime('%b') for i in temp_pivot.columns],
                            y=temp_pivot.index,
                            text=revenue_pivot.values,  # Show revenue values as text
                            texttemplate='$%{text:,.0f}',
                            textfont={"size": 10},
                            colorscale="RdYlBu_r",
                            colorbar=dict(title="Avg Temp (°F)"),
                            hovertemplate='<b>%{y}, %{x}</b><br>' +
                                        'Revenue: $%{text:,.0f}<br>' +
                                        'Avg Temperature: %{z:.1f}°F<extra></extra>'
                        ))
                        fig_rev_temp_heatmap.update_layout(
                            title="Revenue by Day & Month<br><sub>Color: Avg Temperature (°F)</sub>",
                            xaxis_title="Month",
                            yaxis_title="Day of Week",
                            height=450,
                            plot_bgcolor='white',
                            margin=dict(t=70, b=10, l=10, r=10)
                        )
                        st.plotly_chart(fig_rev_temp_heatmap, use_container_width=True)
                    else:
                        st.info("No data for Revenue-Temperature heatmap.")
                
                with temp_row_col2:
                    # Attendance with Temperature Color Scale
                    if attendees_pivot is not None and not attendees_pivot.empty and not temp_pivot.empty:
                        fig_att_temp_heatmap = go.Figure(data=go.Heatmap(
                            z=temp_pivot.values,  # Color based on temperature
                            x=[datetime(2000,i,1).strftime('%b') for i in temp_pivot.columns],
                            y=temp_pivot.index,
                            text=attendees_pivot.values,  # Show attendance values as text
                            texttemplate='%{text:.0f}',
                            textfont={"size": 10},
                            colorscale="RdYlBu_r",
                            colorbar=dict(title="Avg Temp (°F)"),
                            hovertemplate='<b>%{y}, %{x}</b><br>' +
                                        'Avg Attendees: %{text:.0f}<br>' +
                                        'Avg Temperature: %{z:.1f}°F<extra></extra>'
                        ))
                        fig_att_temp_heatmap.update_layout(
                            title="Avg Attendees by Day & Month<br><sub>Color: Avg Temperature (°F)</sub>",
                            xaxis_title="Month",
                            yaxis_title="Day of Week",
                            height=450,
                            plot_bgcolor='white',
                            margin=dict(t=70, b=10, l=10, r=10)
                        )
                        st.plotly_chart(fig_att_temp_heatmap, use_container_width=True)
                    else:
                        st.info("No data for Attendance-Temperature heatmap.")
                
                # Row 2: Precipitation-based heatmaps
                st.markdown("##### Precipitation Impact")
                precip_row_col1, precip_row_col2 = st.columns(2)
                
                with precip_row_col1:
                    # Revenue with Precipitation Color Scale
                    if not revenue_pivot.empty and not precip_pivot.empty:
                        fig_rev_precip_heatmap = go.Figure(data=go.Heatmap(
                            z=precip_pivot.values,  # Color based on precipitation
                            x=[datetime(2000,i,1).strftime('%b') for i in precip_pivot.columns],
                            y=precip_pivot.index,
                            text=revenue_pivot.values,  # Show revenue values as text
                            texttemplate='$%{text:,.0f}',
                            textfont={"size": 10},
                            colorscale="Blues",
                            colorbar=dict(title="Avg Precip (mm)"),
                            hovertemplate='<b>%{y}, %{x}</b><br>' +
                                        'Revenue: $%{text:,.0f}<br>' +
                                        'Avg Precipitation: %{z:.2f} mm<extra></extra>'
                        ))
                        fig_rev_precip_heatmap.update_layout(
                            title="Revenue by Day & Month<br><sub>Color: Avg Precipitation (mm)</sub>",
                            xaxis_title="Month",
                            yaxis_title="Day of Week",
                            height=450,
                            plot_bgcolor='white',
                            margin=dict(t=70, b=10, l=10, r=10)
                        )
                        st.plotly_chart(fig_rev_precip_heatmap, use_container_width=True)
                    else:
                        st.info("No data for Revenue-Precipitation heatmap.")
                
                with precip_row_col2:
                    # Attendance with Precipitation Color Scale
                    if attendees_pivot is not None and not attendees_pivot.empty and not precip_pivot.empty:
                        fig_att_precip_heatmap = go.Figure(data=go.Heatmap(
                            z=precip_pivot.values,  # Color based on precipitation
                            x=[datetime(2000,i,1).strftime('%b') for i in precip_pivot.columns],
                            y=precip_pivot.index,
                            text=attendees_pivot.values,  # Show attendance values as text
                            texttemplate='%{text:.0f}',
                            textfont={"size": 10},
                            colorscale="Blues",
                            colorbar=dict(title="Avg Precip (mm)"),
                            hovertemplate='<b>%{y}, %{x}</b><br>' +
                                        'Avg Attendees: %{text:.0f}<br>' +
                                        'Avg Precipitation: %{z:.2f} mm<extra></extra>'
                        ))
                        fig_att_precip_heatmap.update_layout(
                            title="Avg Attendees by Day & Month<br><sub>Color: Avg Precipitation (mm)</sub>",
                            xaxis_title="Month",
                            yaxis_title="Day of Week",
                            height=450,
                            plot_bgcolor='white',
                            margin=dict(t=70, b=10, l=10, r=10)
                        )
                        st.plotly_chart(fig_att_precip_heatmap, use_container_width=True)
                    else:
                        st.info("No data for Attendance-Precipitation heatmap.")
                
                # Row 3: Wind Speed-based heatmaps
                st.markdown("##### Wind Speed Impact")
                wind_row_col1, wind_row_col2 = st.columns(2)
                
                with wind_row_col1:
                    # Revenue with Wind Speed Color Scale
                    if not revenue_pivot.empty and not wind_pivot.empty:
                        fig_rev_wind_heatmap = go.Figure(data=go.Heatmap(
                            z=wind_pivot.values,  # Color based on wind speed
                            x=[datetime(2000,i,1).strftime('%b') for i in wind_pivot.columns],
                            y=wind_pivot.index,
                            text=revenue_pivot.values,  # Show revenue values as text
                            texttemplate='$%{text:,.0f}',
                            textfont={"size": 10},
                            colorscale="Oranges",
                            colorbar=dict(title="Avg Wind (km/h)"),
                            hovertemplate='<b>%{y}, %{x}</b><br>' +
                                        'Revenue: $%{text:,.0f}<br>' +
                                        'Avg Wind Speed: %{z:.1f} km/h<extra></extra>'
                        ))
                        fig_rev_wind_heatmap.update_layout(
                            title="Revenue by Day & Month<br><sub>Color: Avg Wind Speed (km/h)</sub>",
                            xaxis_title="Month",
                            yaxis_title="Day of Week",
                            height=450,
                            plot_bgcolor='white',
                            margin=dict(t=70, b=10, l=10, r=10)
                        )
                        st.plotly_chart(fig_rev_wind_heatmap, use_container_width=True)
                    else:
                        st.info("No data for Revenue-Wind Speed heatmap.")
                
                with wind_row_col2:
                    # Attendance with Wind Speed Color Scale
                    if attendees_pivot is not None and not attendees_pivot.empty and not wind_pivot.empty:
                        fig_att_wind_heatmap = go.Figure(data=go.Heatmap(
                            z=wind_pivot.values,  # Color based on wind speed
                            x=[datetime(2000,i,1).strftime('%b') for i in wind_pivot.columns],
                            y=wind_pivot.index,
                            text=attendees_pivot.values,  # Show attendance values as text
                            texttemplate='%{text:.0f}',
                            textfont={"size": 10},
                            colorscale="Oranges",
                            colorbar=dict(title="Avg Wind (km/h)"),
                            hovertemplate='<b>%{y}, %{x}</b><br>' +
                                        'Avg Attendees: %{text:.0f}<br>' +
                                        'Avg Wind Speed: %{z:.1f} km/h<extra></extra>'
                        ))
                        fig_att_wind_heatmap.update_layout(
                            title="Avg Attendees by Day & Month<br><sub>Color: Avg Wind Speed (km/h)</sub>",
                            xaxis_title="Month",
                            yaxis_title="Day of Week",
                            height=450,
                            plot_bgcolor='white',
                            margin=dict(t=70, b=10, l=10, r=10)
                        )
                        st.plotly_chart(fig_att_wind_heatmap, use_container_width=True)
                    else:
                        st.info("No data for Attendance-Wind Speed heatmap.")
            else:
                st.warning("StartDate column not available for heatmap generation.")

            st.markdown("<hr style='margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True)

            st.markdown("### Key Performance Indicators: Event Metrics by Weather")
            if 'ActualRevenue' in event_level_for_impact.columns and 'EventID' in event_level_for_impact.columns and 'simple_weather' in event_level_for_impact.columns:
                revenue_by_simple_weather = event_level_for_impact.groupby('simple_weather')['ActualRevenue'].sum().sort_values(ascending=False)
                events_by_simple_weather = event_level_for_impact.groupby('simple_weather')['EventID'].nunique().sort_values(ascending=False)
                kpi_weather_col1, kpi_weather_col2 = st.columns(2)
                with kpi_weather_col1:
                    st.metric(label=f"Total Revenue on Clear/Mainly Clear Days", value=f"${revenue_by_simple_weather.get('Clear/Mainly Clear', 0):,.0f}")
                    st.metric(label=f"Total Revenue on Rainy Days", value=f"${revenue_by_simple_weather.get('Rainy', 0):,.0f}")
                    st.metric(label=f"Total Revenue on Snowy Days", value=f"${revenue_by_simple_weather.get('Snowy', 0):,.0f}")
                with kpi_weather_col2:
                    st.metric(label=f"Events on Clear/Mainly Clear Days", value=f"{events_by_simple_weather.get('Clear/Mainly Clear', 0)}")
                    st.metric(label=f"Events on Rainy Days", value=f"{events_by_simple_weather.get('Rainy', 0)}")
                    st.metric(label=f"Events on Snowy Days", value=f"{events_by_simple_weather.get('Snowy', 0)}")
            else:
                #st.warning("Required columns for weather KPIs (ActualRevenue, EventID, simple_weather) not found.")
                pass

            # st.markdown("<br>", unsafe_allow_html=True)
            # st.markdown("### Event & Revenue Analysis by Detailed Weather Conditions")
            # viz_col1, viz_col2 = st.columns(2)
            # with viz_col1:
            #     if 'EventID' in event_level_for_impact.columns and 'weather_condition' in event_level_for_impact.columns:
            #         event_count_by_cond = event_level_for_impact.groupby('weather_condition')['EventID'].nunique().reset_index(name='EventCount').sort_values(by='EventCount', ascending=False)
            #         fig_event_count_weather = px.bar(event_count_by_cond.head(10), y='weather_condition', x='EventCount', orientation='h', title='Event Count by Weather (Top 10)', labels={'EventCount': 'Number of Events', 'weather_condition': 'Weather'}, text='EventCount')
            #         fig_event_count_weather.update_traces(marker_color='#6495ED', texttemplate='%{text}', textposition='outside')
            #         fig_event_count_weather.update_layout(height=400, yaxis_title=None, plot_bgcolor='white', margin=dict(l=10, r=10, t=30, b=20), yaxis=dict(autorange="reversed"))
            #         st.plotly_chart(fig_event_count_weather, use_container_width=True, key="weather_impact_event_count_chart_v2")
            #     else: st.info("Data missing for 'Event Count by Weather' chart.")
            # with viz_col2:
            #     if 'ActualRevenue' in event_level_for_impact.columns and 'weather_condition' in event_level_for_impact.columns:
            #         avg_revenue_by_cond = event_level_for_impact.groupby('weather_condition')['ActualRevenue'].mean().reset_index().sort_values(by='ActualRevenue', ascending=False)
            #         fig_avg_rev_weather = px.bar(avg_revenue_by_cond.head(10), y='weather_condition', x='ActualRevenue', orientation='h', title='Avg. Event Revenue by Weather (Top 10)', labels={'ActualRevenue': 'Avg. Revenue ($)', 'weather_condition': 'Weather'}, text='ActualRevenue')
            #         fig_avg_rev_weather.update_traces(marker_color='#FF7F50', texttemplate='$%{text:,.0f}', textposition='outside')
            #         fig_avg_rev_weather.update_layout(height=400, yaxis_title=None, plot_bgcolor='white', margin=dict(l=10, r=10, t=30, b=20), yaxis=dict(autorange="reversed"))
            #         st.plotly_chart(fig_avg_rev_weather, use_container_width=True, key="weather_impact_avg_rev_chart_v2")
            #     else: st.info("Data missing for 'Avg. Event Revenue by Weather' chart.")

            st.markdown("<hr style='margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True)
            st.markdown("### Analysis by Weather Variable Categories")

            # Prepare data for daily metrics to be binned (now using Fahrenheit)
            analysis_df['event_date'] = pd.to_datetime(analysis_df['StartDate']).dt.date
            daily_event_metrics_for_binning = analysis_df.groupby('event_date').agg(
                TotalDailyRevenue=('ActualRevenue', 'sum'),
                TotalDailyEvents=('EventID', 'nunique')
            ).reset_index()
            daily_weather_vars_for_binning = analysis_df[['event_date', 'temperature_2m_max_f', 'precipitation_sum', 'windspeed_10m_max', 'daylight_time']].drop_duplicates(subset=['event_date'], keep='first')
            binned_analysis_df = pd.merge(daily_event_metrics_for_binning, daily_weather_vars_for_binning, on='event_date', how='left')
            binned_analysis_df.dropna(subset=['temperature_2m_max_f', 'windspeed_10m_max', 'daylight_time', 'precipitation_sum'], inplace=True) # Ensure no NaN for binning columns

            if not binned_analysis_df.empty:
                # Temperature Bins (now in Fahrenheit)
                temp_bins = [-np.inf, 32, 50, 68, 86, np.inf]  # Fahrenheit: <32°F, 32-50°F, 50-68°F, 68-86°F, >86°F
                temp_labels = ['<32°F', '32-50°F', '50-68°F', '68-86°F', '>86°F']
                binned_analysis_df['temp_bin'] = pd.cut(binned_analysis_df['temperature_2m_max_f'], bins=temp_bins, labels=temp_labels, right=False)
                
                # Wind Speed Bins
                wind_bins = [0, 15, 30, 45, np.inf] # km/h example bins
                wind_labels = ['Light (0-15)', 'Moderate (15-30)', 'Strong (30-45)', 'Very Strong (>45)']
                binned_analysis_df['wind_bin'] = pd.cut(binned_analysis_df['windspeed_10m_max'], bins=wind_bins, labels=wind_labels, right=False)

                # Daylight Hours Bins
                daylight_bins = [0, 9, 12, 15, np.inf] # hours example bins
                daylight_labels = ['<9h', '9-12h', '12-15h', '>15h']
                binned_analysis_df['daylight_bin'] = pd.cut(binned_analysis_df['daylight_time'], bins=daylight_bins, labels=daylight_labels, right=False)
                
                # Precipitation Bins
                precip_bins = [-np.inf, 0.1, 2.5, 10, np.inf] # mm, 0 for no rain, then light, moderate, heavy
                precip_labels = ['No Precip', 'Light (0.1-2.5mm)', 'Moderate (2.5-10mm)', 'Heavy (>10mm)']
                binned_analysis_df['precip_bin'] = pd.cut(binned_analysis_df['precipitation_sum'], bins=precip_bins, labels=precip_labels, right=True, include_lowest=True)

                # --- Plotting binned data ---
                st.subheader("Daily Event Metrics by Temperature Range (°F)")
                corr_temp_events = binned_analysis_df['temperature_2m_max_f'].corr(binned_analysis_df['TotalDailyEvents'])
                corr_temp_revenue = binned_analysis_df['temperature_2m_max_f'].corr(binned_analysis_df['TotalDailyRevenue'])
                bin_col1, bin_col2 = st.columns(2)
                with bin_col1:
                    avg_events_by_temp = binned_analysis_df.groupby('temp_bin', observed=False)['TotalDailyEvents'].mean().reset_index()
                    fig_events_temp_bin = px.bar(avg_events_by_temp, x='temp_bin', y='TotalDailyEvents', title=f'Avg Daily Events by Temp (Corr: {corr_temp_events:.2f})', labels={'temp_bin': 'Temperature Range', 'TotalDailyEvents': 'Avg Daily Events'}, color='temp_bin')
                    st.plotly_chart(fig_events_temp_bin, use_container_width=True, key="events_temp_bin_chart")
                with bin_col2:
                    avg_revenue_by_temp = binned_analysis_df.groupby('temp_bin', observed=False)['TotalDailyRevenue'].mean().reset_index()
                    fig_revenue_temp_bin = px.bar(avg_revenue_by_temp, x='temp_bin', y='TotalDailyRevenue', title=f'Avg Daily Revenue by Temp (Corr: {corr_temp_revenue:.2f})', labels={'temp_bin': 'Temperature Range', 'TotalDailyRevenue': 'Avg Daily Revenue ($'}, color='temp_bin')
                    st.plotly_chart(fig_revenue_temp_bin, use_container_width=True, key="revenue_temp_bin_chart")

                st.subheader("Daily Event Metrics by Wind Speed Range")
                corr_wind_events = binned_analysis_df['windspeed_10m_max'].corr(binned_analysis_df['TotalDailyEvents'])
                corr_wind_revenue = binned_analysis_df['windspeed_10m_max'].corr(binned_analysis_df['TotalDailyRevenue'])
                bin_col3, bin_col4 = st.columns(2)
                with bin_col3:
                    avg_events_by_wind = binned_analysis_df.groupby('wind_bin', observed=False)['TotalDailyEvents'].mean().reset_index()
                    fig_events_wind_bin = px.bar(avg_events_by_wind, x='wind_bin', y='TotalDailyEvents', title=f'Avg Daily Events by Wind (Corr: {corr_wind_events:.2f})', labels={'wind_bin': 'Wind Speed Range', 'TotalDailyEvents': 'Avg Daily Events'}, color='wind_bin')
                    st.plotly_chart(fig_events_wind_bin, use_container_width=True, key="events_wind_bin_chart")
                with bin_col4:
                    avg_revenue_by_wind = binned_analysis_df.groupby('wind_bin', observed=False)['TotalDailyRevenue'].mean().reset_index()
                    fig_revenue_wind_bin = px.bar(avg_revenue_by_wind, x='wind_bin', y='TotalDailyRevenue', title=f'Avg Daily Revenue by Wind (Corr: {corr_wind_revenue:.2f})', labels={'wind_bin': 'Wind Speed Range', 'TotalDailyRevenue': 'Avg Daily Revenue ($'}, color='wind_bin')
                    st.plotly_chart(fig_revenue_wind_bin, use_container_width=True, key="revenue_wind_bin_chart")
                
                st.subheader("Daily Event Metrics by Precipitation Range")
                corr_precip_events = binned_analysis_df['precipitation_sum'].corr(binned_analysis_df['TotalDailyEvents'])
                corr_precip_revenue = binned_analysis_df['precipitation_sum'].corr(binned_analysis_df['TotalDailyRevenue'])
                bin_col_precip1, bin_col_precip2 = st.columns(2)
                with bin_col_precip1:
                    avg_events_by_precip = binned_analysis_df.groupby('precip_bin', observed=False)['TotalDailyEvents'].mean().reset_index()
                    fig_events_precip_bin = px.bar(avg_events_by_precip, x='precip_bin', y='TotalDailyEvents', title=f'Avg Daily Events by Precip. (Corr: {corr_precip_events:.2f})', labels={'precip_bin': 'Precipitation Range', 'TotalDailyEvents': 'Avg Daily Events'}, color='precip_bin')
                    st.plotly_chart(fig_events_precip_bin, use_container_width=True, key="events_precip_bin_chart")
                with bin_col_precip2:
                    avg_revenue_by_precip = binned_analysis_df.groupby('precip_bin', observed=False)['TotalDailyRevenue'].mean().reset_index()
                    fig_revenue_precip_bin = px.bar(avg_revenue_by_precip, x='precip_bin', y='TotalDailyRevenue', title=f'Avg Daily Revenue by Precip. (Corr: {corr_precip_revenue:.2f})', labels={'precip_bin': 'Precipitation Range', 'TotalDailyRevenue': 'Avg Daily Revenue ($'}, color='precip_bin')
                    st.plotly_chart(fig_revenue_precip_bin, use_container_width=True, key="revenue_precip_bin_chart")

                st.subheader("Daily Event Metrics by Daylight Hours")
                corr_daylight_events = binned_analysis_df['daylight_time'].corr(binned_analysis_df['TotalDailyEvents'])
                corr_daylight_revenue = binned_analysis_df['daylight_time'].corr(binned_analysis_df['TotalDailyRevenue'])
                bin_col5, bin_col6 = st.columns(2)
                with bin_col5:
                    avg_events_by_daylight = binned_analysis_df.groupby('daylight_bin', observed=False)['TotalDailyEvents'].mean().reset_index()
                    fig_events_daylight_bin = px.bar(avg_events_by_daylight, x='daylight_bin', y='TotalDailyEvents', title=f'Avg Daily Events by Daylight (Corr: {corr_daylight_events:.2f})', labels={'daylight_bin': 'Daylight Hours', 'TotalDailyEvents': 'Avg Daily Events'}, color='daylight_bin')
                    st.plotly_chart(fig_events_daylight_bin, use_container_width=True, key="events_daylight_bin_chart")
                with bin_col6:
                    avg_revenue_by_daylight = binned_analysis_df.groupby('daylight_bin', observed=False)['TotalDailyRevenue'].mean().reset_index()
                    fig_revenue_daylight_bin = px.bar(avg_revenue_by_daylight, x='daylight_bin', y='TotalDailyRevenue', title=f'Avg Daily Revenue by Daylight (Corr: {corr_daylight_revenue:.2f})', labels={'daylight_bin': 'Daylight Hours', 'TotalDailyRevenue': 'Avg Daily Revenue ($'}, color='daylight_bin')
                    st.plotly_chart(fig_revenue_daylight_bin, use_container_width=True, key="revenue_daylight_bin_chart")
            else:
                st.info("Not enough aggregated daily data for binned analysis.")
            
            st.markdown("<hr style='margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True)
            
            # Add merged data table at the end of the page
            st.subheader("Merged Event and Weather Data")
            
            if not df_merged_with_weather.empty:
                # Select relevant columns for display (including Fahrenheit temperatures)
                display_columns = [
                    'EventID', 'Event_Description', 'StartDate', 'OrderedAttendance', 'ActualRevenue',
                    'temperature_2m_max_f', 'temperature_2m_min_f', 'precipitation_sum', 'rain_sum', 
                    'snowfall_sum', 'windspeed_10m_max', 'weathercode', 'weather_condition', 'daylight_time'
                ]
                
                # Filter to only columns that exist in the dataframe
                available_columns = [col for col in display_columns if col in df_merged_with_weather.columns]
                
                # Add Fahrenheit columns to the merged display dataframe if they don't exist
                df_merged_display_copy = df_merged_with_weather.copy()
                if 'temperature_2m_max' in df_merged_display_copy.columns and 'temperature_2m_max_f' not in df_merged_display_copy.columns:
                    df_merged_display_copy['temperature_2m_max_f'] = df_merged_display_copy['temperature_2m_max'].apply(celsius_to_fahrenheit)
                if 'temperature_2m_min' in df_merged_display_copy.columns and 'temperature_2m_min_f' not in df_merged_display_copy.columns:
                    df_merged_display_copy['temperature_2m_min_f'] = df_merged_display_copy['temperature_2m_min'].apply(celsius_to_fahrenheit)
                
                # Update available columns list
                available_columns = [col for col in display_columns if col in df_merged_display_copy.columns]
                
                if available_columns:
                    # Create a copy with only the selected columns
                    merged_display_df = df_merged_display_copy[available_columns].copy()
                    
                    # Format the dates and numeric columns
                    if 'StartDate' in merged_display_df.columns:
                        merged_display_df['StartDate'] = pd.to_datetime(merged_display_df['StartDate']).dt.strftime('%Y-%m-%d')
                    
                    if 'ActualRevenue' in merged_display_df.columns:
                        merged_display_df['ActualRevenue'] = merged_display_df['ActualRevenue'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
                    
                    # Format temperature columns
                    if 'temperature_2m_max_f' in merged_display_df.columns:
                        merged_display_df['temperature_2m_max_f'] = merged_display_df['temperature_2m_max_f'].apply(lambda x: f"{x:.1f}°F" if pd.notna(x) else "")
                    if 'temperature_2m_min_f' in merged_display_df.columns:
                        merged_display_df['temperature_2m_min_f'] = merged_display_df['temperature_2m_min_f'].apply(lambda x: f"{x:.1f}°F" if pd.notna(x) else "")
                    
                    # Add a toggle to show/hide the data table
                    show_data = st.checkbox("Show merged event and weather data table", value=False)
                    
                    if show_data:
                        st.dataframe(merged_display_df, use_container_width=True)
                        
                        # Add download button for the data
                        csv = df_merged_display_copy[available_columns].to_csv(index=False)
                        st.download_button(
                            label="Download Data as CSV",
                            data=csv,
                            file_name="event_weather_data.csv",
                            mime="text/csv",
                        )
                else:
                    st.warning("No common columns found between the merged dataframe and the selected display columns.")
            else:
                st.warning("No merged weather and event data available to display.")

            st.markdown("<br>", unsafe_allow_html=True)

            # # Event & Revenue Analysis by Detailed Weather Conditions - moved to top
            # st.markdown("### Event & Revenue Analysis by Detailed Weather Conditions")
            # viz_col1_top, viz_col2_top = st.columns(2)
            # with viz_col1_top:
            #     if 'EventID' in event_level_for_impact.columns and 'weather_condition' in event_level_for_impact.columns:
            #         event_count_by_cond = event_level_for_impact.groupby('weather_condition')['EventID'].nunique().reset_index(name='EventCount').sort_values(by='EventCount', ascending=False)
            #         fig_event_count_weather = px.bar(event_count_by_cond.head(10), y='weather_condition', x='EventCount', orientation='h', title='Event Count by Weather (Top 10)', labels={'EventCount': 'Number of Events', 'weather_condition': 'Weather'}, text='EventCount')
            #         fig_event_count_weather.update_traces(marker_color='#6495ED', texttemplate='%{text}', textposition='outside')
            #         fig_event_count_weather.update_layout(height=400, yaxis_title=None, plot_bgcolor='white', margin=dict(l=10, r=10, t=30, b=20), yaxis=dict(autorange="reversed"))
            #         st.plotly_chart(fig_event_count_weather, use_container_width=True, key="weather_impact_event_count_chart_top")
            #     else: st.info("Data missing for 'Event Count by Weather' chart.")
            # with viz_col2_top:
            #     if 'ActualRevenue' in event_level_for_impact.columns and 'weather_condition' in event_level_for_impact.columns:
            #         avg_revenue_by_cond = event_level_for_impact.groupby('weather_condition')['ActualRevenue'].mean().reset_index().sort_values(by='ActualRevenue', ascending=False)
            #         fig_avg_rev_weather = px.bar(avg_revenue_by_cond.head(10), y='weather_condition', x='ActualRevenue', orientation='h', title='Avg. Event Revenue by Weather (Top 10)', labels={'ActualRevenue': 'Avg. Revenue ($)', 'weather_condition': 'Weather'}, text='ActualRevenue')
            #         fig_avg_rev_weather.update_traces(marker_color='#FF7F50', texttemplate='$%{text:,.0f}', textposition='outside')
            #         fig_avg_rev_weather.update_layout(height=400, yaxis_title=None, plot_bgcolor='white', margin=dict(l=10, r=10, t=30, b=20), yaxis=dict(autorange="reversed"))
            #         st.plotly_chart(fig_avg_rev_weather, use_container_width=True, key="weather_impact_avg_rev_chart_top")
            #     else: st.info("Data missing for 'Avg. Event Revenue by Weather' chart.")

            # st.markdown("<hr style='margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True)

            # NEW: Time Series Analysis Section
