import streamlit as st
import pandas as pd
upload_file = st.file_uploader('アクセスログをアップロードしてください。')

if upload_file is not None:
    df = pd.read_csv(upload_file)
    st.markdown('### アクセスログ(先頭5件)')
    st.write(df.head())
