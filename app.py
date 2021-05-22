
"""Home page shown when users enter the application"""

import streamlit as st
import pages.overview
import pages.example
import pandas as pd
import numpy as np





PAGES = {
    "Overview": pages.overview,
    "Example": pages.example
}

def main():
    """Main function of the App"""
    pl = st.empty()
    pl.markdown('''
    <html>
    <body style="background-color:#EB3A42;">
    <h1 align="center" style="color:white;">SumSwift Demo V0.1</h1>
    </body>
    </html>
    ''',unsafe_allow_html=True)
    # st.image('unpackai.jpeg',width=220)
    pl2 = st.empty()
    pl2.markdown('''
    <marquee style='width: 100%; color: black;	
    font-family: "Microsoft YaHei", Arial, serif;'>
    <b>Text Summarization for Chinese</b>
    </marquee>
    ''',unsafe_allow_html=True)
    password = st.sidebar.text_input('Please enter your password','')
    if password =='sumswift':
        pl2.empty()
        # place_holder.empty()
        # df = load_data(file_ID)
        st.sidebar.subheader("Navigation")
        selection = st.sidebar.radio("", list(PAGES.keys()))
        page = PAGES[selection]
        with st.spinner(f'Loading {selection} ...'):
            # pl.empty()
            page.write()
        
        st.sidebar.title("About")
        st.sidebar.info(
            """
            This demo is for Demonstrating sumswift functions
        """
            )
    elif password =='':
        pass
    else:
        st.sidebar.warning('Incorrect password')

if __name__ == "__main__":
    main()