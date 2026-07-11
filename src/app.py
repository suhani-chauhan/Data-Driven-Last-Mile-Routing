"""Entry point for the multi-page dashboard. Pure navigation/structure --
all the route-solving demo logic lives unchanged in pages/home.py (it was
simply moved here from what used to be this file's own body); pages/about.py
and pages/how_to_use.py are plain informational pages.

Launch:
    streamlit run src/app.py
"""
import streamlit as st

pg = st.navigation(
    [
        st.Page("pages/home.py", title="Home", icon="🏠", default=True),
        st.Page("pages/about.py", title="About", icon="ℹ️"),
        st.Page("pages/how_to_use.py", title="How to Use", icon="❓"),
    ],
    position="top",
)
pg.run()
