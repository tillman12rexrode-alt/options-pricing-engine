"""Options Pricing Engine - a browser app over the optpricing library.

This file only sets up the page and the navigation. Each page lives in its own
module: stock_options.py for single stocks' chains, research.py for their
filings and term structure, index_options.py for the index surface scanner. Keeping the entrypoint this thin is what lets each page be
tested on its own.

Run it with:

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st
import theme

st.set_page_config(page_title="Options Pricing Engine", page_icon="📈", layout="wide")
theme.apply()

st.navigation(
    [
        st.Page("views/stocks.py", title="Stock options", url_path="stocks", default=True),
        st.Page("views/research.py", title="Stock research", url_path="research"),
        st.Page("views/index.py", title="Index options", url_path="index"),
    ]
).run()
