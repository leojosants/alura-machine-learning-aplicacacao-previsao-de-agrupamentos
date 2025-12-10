
import streamlit as st__streamlit
import pandas as pd__pandas
import joblib as joblib


encoder = joblib.load("./2-bases-de-dados-tratadas/encoder.pkl")
scaler = joblib.load("./2-bases-de-dados-tratadas/scaler.pkl")
kmeans = joblib.load("./2-bases-de-dados-tratadas/kmeans.pkl")

st__streamlit.title("Grupos de interesse para marketing")

st__streamlit.write("""
    Neste projeto, aplicamos o algoritmo de clusterização K-means para identificar e prever agrupamentos de interesses de usuários, com o objetivo de direcionar campanhas de marketing de forma mais eficaz.
    Através dessa análise, conseguimos segmentar o público em bolhas de interesse, permitindo a criação de campanhas personalizadas e mais assertivas, com base nos padrões de comportamento e preferências de cada grupo.
""")

up_file =st__streamlit.file_uploader(
    "Escolha um arquivo 'CSV' para realizar a previsão", 
    type="csv"
)

def processar_prever(df__dataframe):
    encoded_sexo = encoder.transform(df__dataframe[["sexo"]])
    encoded_df__dataframe = pd__pandas.DataFrame(encoded_sexo, columns=encoder.get_feature_names_out(["sexo"]))
    dados = pd__pandas.concat([df__dataframe.drop("sexo", axis=1), encoded_df__dataframe], axis=1)

    dados_escalados = scaler.transform(dados)

    cluster = kmeans.predict(dados_escalados)

    return cluster

if up_file is not None:
    st__streamlit.write("""
        ### Descrição dos Grupos:
        - **Grupo 0** é focado em um público jovem com forte interesse em moda, música e aparência.
        - **Grupo 1** está muito associado a esportes, especialmente futebol americano, basquete e atividades culturais como banda e rock.
        - **Grupo 2** é mais equilibrado, com interesses em música, dança, e moda.
    """)

    df__dataframe = pd__pandas.read_csv(up_file)
    cluster = processar_prever(df__dataframe)
    df__dataframe.insert(0, "grupos", cluster)

    st__streamlit.write("Visualização dos resultados (10 primeiros registros)")
    st__streamlit.write(df__dataframe.head(10))

    csv = df__dataframe.to_csv(index=False)

    st__streamlit.download_button(
        label="Baixar resultados completos", 
        data=csv, 
        file_name="grupos_interesse.csv", 
        mime="text/csv"
    )