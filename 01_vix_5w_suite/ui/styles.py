import streamlit as st

def setup_page_config():
    st.set_page_config(
        page_title="VIX 5% Weekly Suite",
        layout="wide",
        initial_sidebar_state="expanded",
    )

def inject_global_css():
    st.markdown(
        '''
        <style>
        .main { background-color: #111111; }
        .stMetric-label, .stMarkdown, .stTextInput label, .stSelectbox label {
            color: #dddddd !important;
        }
        </style>
        ''',
        unsafe_allow_html=True,
    )
