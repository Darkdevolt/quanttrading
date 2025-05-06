import pdfplumber

with pdfplumber.open(uploaded_file) as pdf:
    first_page = pdf.pages[0]
    table = first_page.extract_table()
    df = pd.DataFrame(table[1:], columns=table[0])
    st.dataframe(df)
