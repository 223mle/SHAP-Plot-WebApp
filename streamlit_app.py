import streamlit as st
import pandas as pd
upload_file = st.file_uploader('解析したいcsvファイルをアップロードしてください.')

# icon設定
st.set_page_config(page_title='ozro_wepapp',
                    page_icon='clubhouse-icon.png')


if upload_file is not None:
    df = pd.read_csv(upload_file)
    st.markdown('### データの先頭5件')
    st.write(df.head())

    unuse_cols = st.multiselect(
        '解析不要な列があれば選択してください.',
        list(len(df.columns))
    )
    if len(unuse_cols)==0:
        st.error('解析する列がありません.')

