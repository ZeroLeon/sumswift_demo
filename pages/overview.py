import streamlit as st


def write():
    """Used to write the page in the app.py file"""
    st.header('Text Summarization for Chinese')
    st.write(
            """
### Dataset used for training this model:
[nlpcc 2017](http://tcci.ccf.org.cn/conference/2017/taskdata.php) Single Document Summarization
    """
        )

if __name__ == "__main__":
    write()