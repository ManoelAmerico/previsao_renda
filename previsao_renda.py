import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import joblib

from sklearn.preprocessing import LabelEncoder
import streamlit as st

st.set_page_config(page_title = 'Previsão de renda',
    layout="wide",
    initial_sidebar_state='expanded')


def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def load_model():
    return joblib.load("modelo_renda.pkl")

def sexo_tratado(df):
    df = LabelEncoder().fit_transform(df['sexo'])
    return df

def main():
    st.markdown("---")

    st.sidebar.title("Carregue seu DataFrame")
    uploaded_file = st.sidebar.file_uploader("Faça upload de um arquivo CSV:", type=["csv"])
    data_file_1 = "./input/previsao_de_renda.csv"


    if (data_file_1 is not None):
        df = pd.read_csv(data_file_1, infer_datetime_format=True, parse_dates=['data_ref'])

        st.title("Data de inicio e fim")
        data_inicial = st.date_input("Selecione a data inicial:", df["data_ref"].min().date())
        data_final = st.date_input("Selecione a data final:", df["data_ref"].max().date())
        data_inicial = pd.to_datetime(data_inicial).normalize()
        data_final = pd.to_datetime(data_final).normalize()
        df = df[(df["data_ref"] >= data_inicial) & (df["data_ref"] <= data_final)]

        st.write(f"Exibindo dados de {data_inicial.date()} até {data_final.date()}")

        metadados = pd.DataFrame({'dtypes': df.dtypes})
        metadados['missing'] = df.isna().sum()
        metadados['perc_missing'] = round((metadados['missing']/df.shape[0])*100)
        metadados['valores_unicos'] = df.nunique()

        col1, col2 = st.columns(2)

        with col1:
            mostrar_df = st.checkbox("Exibir DataFrame", value=True)
            if mostrar_df:
                st.subheader("DataFrame")
                st.dataframe(df)

        with col2:
            mostrar_metadados = st.checkbox("Exibir Metadados", value=True)
            if mostrar_metadados:
                st.subheader("Metadados")
                metadados

        st.write('---')

        col3, col4 = st.columns(2)
        plt.figure(figsize=(6,6))

        with col3:
            st.subheader("Análise Univariada")
            var_uni = st.selectbox("Selecione a variável para análise univariada:", df.columns.drop(['id_cliente', 'Unnamed: 0', 'data_ref']))
            
            fig_uni, ax_uni = plt.subplots()
            if df[var_uni].dtype == "object":
                sns.countplot(x=var_uni, data=df, ax=ax_uni)
            else:
                sns.histplot(df[var_uni], kde=True, ax=ax_uni)
            st.pyplot(fig_uni)

        with col4:
            st.subheader("Análise Bivariada")
            var_bi2 = st.selectbox("Selecione a segunda variável:", df.columns.drop(['id_cliente', 'Unnamed: 0', 'data_ref']))

            fig_bi, ax_bi = plt.subplots()
            if df[var_uni].dtype == "object" or df[var_bi2].dtype == "object":
                sns.boxplot(x=var_uni, y=var_bi2, data=df, ax=ax_bi)
            else:
                sns.scatterplot(x=var_uni, y=var_bi2, data=df, ax=ax_bi)
            st.pyplot(fig_bi)

        st.write("----")
        st.title("Distribuição de Variáveis ao Longo do Tempo")

        variavel = st.selectbox("Selecione a variável:", df.columns.drop(['id_cliente', 'Unnamed: 0', 'data_ref']))

        if df[variavel].dtypes == "float64" or "int64":
            st.subheader(f"Distribuição da variável '{variavel}' ao longo do tempo")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(x="data_ref", y=variavel, data=df, ax=ax)
            ax.set_title(f"Tendência de {variavel} ao longo do tempo")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        elif df[variavel].dtypes == "object" or "bool":
            st.subheader(f"Distribuição da variável '{variavel}' ao longo do tempo")
            if df[variavel].dtype == "object":
                df_counts = df.groupby(["data_ref", variavel]).size().reset_index(name="Contagem")
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.histplot(data=df_counts, x="Data", weights="Contagem", hue=variavel, multiple="stack", ax=ax)
                ax.set_title(f"Distribuição de {variavel} ao longo do tempo")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.error("Selecione uma variável qualitativa válida.")
                
        st.write("----")

        st.title("Gráfico de Estabilidade")
        variavel = st.selectbox("Escolha a variável para o gráfico de estabilidade:", df.columns.drop(['id_cliente', 'Unnamed: 0', 'data_ref']))

        if df[variavel].dtype in ["int64", "float64"]: 
            st.subheader(f"Gráfico de Estabilidade para {variavel}")

            media = df[variavel].mean()
            std = df[variavel].std()
            limite_superior = media + 2 * std
            limite_inferior = media - 2 * std

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df["data_ref"],
                y=df[variavel],
                mode="lines+markers",
                name="Valores",
                line=dict(color="blue"),
                marker=dict(size=5)
            ))

            fig.add_trace(go.Scatter(
                x=df["data_ref"],
                y=[media] * len(df),
                mode="lines",
                name="Média",
                line=dict(color="green", dash="dash")
            ))

            fig.add_trace(go.Scatter(
                x=df["data_ref"],
                y=[limite_superior] * len(df),
                mode="lines",
                name="Limite Superior",
                line=dict(color="red", dash="dot")
            ))
            fig.add_trace(go.Scatter(
                x=df["data_ref"],
                y=[limite_inferior] * len(df),
                mode="lines",
                name="Limite Inferior",
                line=dict(color="red", dash="dot")
            ))

            fig.update_layout(
                title=f"Gráfico de Estabilidade: {variavel}",
                xaxis_title="data_ref",
                yaxis_title=variavel,
                template="plotly_white",
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)

        elif df[variavel].dtype == "object":
            st.subheader(f"Gráfico de Estabilidade para {variavel}")

            df_grouped = df.groupby(["data_ref", variavel]).size().reset_index(name="Contagem")

            fig = go.Figure()

            for categoria in df[variavel].unique():
                subset = df_grouped[df_grouped[variavel] == categoria]
                fig.add_trace(go.Bar(
                    x=subset["data_ref"],
                    y=subset["Contagem"],
                    name=str(categoria)
                ))

            fig.update_layout(
                title=f"Distribuição das Categorias ao Longo do Tempo: {variavel}",
                xaxis_title="data_ref",
                yaxis_title="Contagem",
                barmode="stack",
                template="plotly_white",
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("A variável selecionada não é válida para este gráfico. Escolha uma quantitativa ou categórica.")

        st.write("----")

        st.title("Predição com Modelo de Regressão")
    
        st.write("Este aplicativo realiza predições usando um modelo de regressão.")

        modelo = load_model()
        
        col5, col6 = st.columns(2)

        if uploaded_file is not None:
            df_predicao = pd.read_csv(uploaded_file)

            with col5:
                mostrar_df = st.checkbox("DataFrame usado para construção do modelo", value=True)
                st.dataframe(df)

            with col6:
                mostrar_metadados = st.checkbox("DataFrame para predição", value=True)
                st.dataframe(df_predicao)
            
            df_predicao['sexo'] = sexo_tratado(df_predicao)

            colunas_necessarias = modelo.feature_names_in_  
            if all(col in df_predicao.columns for col in colunas_necessarias):
                if st.sidebar.button("Prever"):
                    predicoes = modelo.predict(df_predicao[colunas_necessarias])
                    df_predicao["Predição"] = predicoes

                    st.write("### DataFrame com Predições:")
                    st.dataframe(df_predicao)

                    csv_download = df_predicao.to_csv(index=False)
                    st.download_button(
                        label="Baixar Resultado com Predições",
                        data=csv_download,
                        file_name="resultado_com_predicoes.csv",
                        mime="text/csv",
                    )
            else:
                st.error(f"O DataFrame deve conter as colunas: {', '.join(colunas_necessarias)}")
        else:
            st.info("Por favor, faça upload de um arquivo CSV para continuar.")

if __name__ == '__main__':
    main()