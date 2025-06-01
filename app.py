# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from datasets import load_dataset
import requests
import plotly.graph_objects as go  
import streamlit as st

import joblib
from sklearn.preprocessing import LabelEncoder

# Carregando o modelo salvo
modelo = joblib.load('modelo_pipeline.pkl')

# Configurações da página
st.set_page_config(
    page_title="Relatório de Inadimplência ",
    page_icon="📊",
    layout="wide"
)

# Carregar Dataset
@st.cache_data
def carregar_dados():
    anos = ['2020', '2021', '2022', '2023', '2024']
    df_list = []

    for ano in anos:
        dataset = load_dataset(
            "Andrea1120/Teste1",
            data_dir=f"planilha 5 anos/{ano}" #Colocar aqui o nome da pasta onde estão os arquivos por ano
        )
        df = dataset['train'].to_pandas()
        df['ano'] = ano  
        df_list.append(df)

    return pd.concat(df_list, ignore_index=True)

df = carregar_dados()

# Aplicar filtros
df_filtrado = df

# Sidebar de navegação
st.sidebar.title("Navegação")
pagina = st.sidebar.radio("Ir para:", ["🏠 Home", "📊 Score de Inadimplência"])

# Página principal
if pagina == "🏠 Home":
    # Título
    st.title("Inadimplência de Crédito no Brasil em 2024 💳")
    #Explicação sobre a inadimplência
    with st.expander("📌 O que é Inadimplência?"):
        st.write("""
        Inadimplência acontece quando uma pessoa ou empresa deixa de pagar uma dívida no prazo combinado com o credor.

        Isso pode envolver parcelas atrasadas de empréstimos, financiamentos ou outras obrigações de crédito. 

        No contexto financeiro, a inadimplência é um indicador importante de risco e saúde econômica — tanto para instituições quanto para o país como um todo.
        """)


    # Explicação inicial
    with st.expander("✨ O que é este relatório?"):
        st.write("""
        Este dashboard tem como objetivo analisar, de forma exploratória, os padrões e fatores que influenciam a inadimplência de crédito no Brasil.

        A partir de uma base de dados do Banco Central, buscamos entender *quem são os inadimplentes, **onde estão localizados, **quais tipos de crédito apresentam maior risco* e *quais perfis estão mais associados ao comportamento de inadimplência*.

        A visualização está organizada por *UF, **região, **modalidade de crédito, **porte* e *tipo de cliente*, oferecendo uma visão acessível, clara e interativa para facilitar a tomada de decisões e gerar insights.
        """)
    # Explicação inicial
    with st.expander("Metodologia"):
        st.write("""
        Como metologia para o desenvolvimento deste projeto foi utilizado o **K**nowledge Discovery in Databases (KDD) que é um processo de descoberta de conhecimento em grandes volumes de dados. O KDD envolve várias etapas, desde a seleção e pré-processamento dos dados até a análise e interpretação dos resultados.
        O processo KDD é frequentemente dividido em várias etapas principais:
        - **Seleção de Dados**: Utilização de dados do Banco Central do Brasil, abrangendo informações sobre inadimplência de crédito em 2024.
        - **Pré-processamento**: 
                 - Remoção de valores ausentes: Excluímos linhas com dados faltantes e colunas totalmente vazias para garantir consistência.
                 - Eliminação de duplicatas: Removemos linhas repetidas para evitar distorções nas análises.
                 - Descarte de colunas irrelevantes: Excluímos colunas que não seriam utilizadas nas análises (cnae_secao, cnae_subclasse, numero_de_operacoes).Descarte de colunas irrelevantes: Excluímos colunas que não seriam utilizadas nas análises (cnae_secao, cnae_subclasse, numero_de_operacoes).
                 - Padronização dos nomes de colunas: Espaços extras foram removidos para evitar erros na manipulação dos dados.
                 - Visualização de valores únicos: Percorremos as colunas para verificar possíveis inconsistências e facilitar futuras limpezas.
                 - Unificação e particionamento dos arquivos: Todos os CSVs foram combinados e particionados por mês no PySpark, otimizando o desempenho em grandes volumes de dados.
                 - Salvamento eficiente: Dados armazenados nos formatos Parquet e CSV para facilitar o acesso e análise.
        - **Transformação**:   
                 - Conversão de dados: Colunas categóricas foram convertidas em numéricas para facilitar a análise.
                 - Criação de novas variáveis: Variáveis como "inadimplente" foram criadas para facilitar a análise.
                 - Normalização: Dados foram normalizados para garantir que todas as variáveis estivessem na mesma escala.
                 - Divisão de dados: O conjunto de dados foi dividido em conjuntos de treinamento e teste para validação do modelo.
        - **Mineração de Dados**: Realização de Análise Explortória e utilização de técnicas de aprendizado de máquina, como o Random Forest, para prever a probabilidade de inadimplência com base em variáveis como "porte", "ocupação", "modalidade" e "total a vencer". 
        - **Avaliação**: Avaliação do desempenho do modelo utilizando métricas como AUC.
        - **Implementação**: Utilização do modelo em um aplicativo web interativo, permitindo que os usuários insiram dados e recebam previsões de inadimplência.
        """)

    with st.expander("📚 Glossário de Variáveis"):
        st.markdown("""
        - *data_base*: Data de referência da informação financeira.
        - *UF*: Unidade da Federação (estado).
        - *região*: Região geográfica do Brasil (Norte, Sul, etc.).
        - *TCB*: Tipo de Crédito Bancário (Bancário, Não Bancário, Cooperativas).
        - *SR*: Segmento de Risco (S1 a S5, onde S1 é menor risco e S5 maior).
        - *cliente*: Tipo de cliente (PF - Pessoa Física, PJ - Pessoa Jurídica).
        - *ocupação*: Categoria de atividade econômica do cliente.
        - *porte*: Porte do cliente (ex: Micro, Pequeno, Médio, Grande).
        - *modalidade*: Tipo de operação de crédito (ex: Cartão de Crédito, Financiamento, Empréstimo Pessoal).
        - *origem*: Indica se o crédito tem finalidade específica (ex: imobiliário, educacional).
        - *indexador*: Tipo de taxa aplicada ao crédito (ex: prefixado, pós-fixado, atrelado à inflação).
        - *vencido_acima_de_15_dias*: Valor de parcelas em atraso por mais de 15 dias.
        - *carteira_ativa*: Valor total de créditos ainda em vigor.
        - *carteira_inadimplida_arrastada*: Total de valores inadimplentes que foram mantidos na carteira.
        - *ativo_problematico*: Soma de créditos com risco elevado ou inadimplência.
        - *mes_texto*: Nome do mês correspondente à data da base.
        """)

    # Filtros no topo
    st.subheader("🎯 Filtros")

    col_f1, col_f2, col_f3 = st.columns(3)
    col_f4, col_f5, col_f6 = st.columns(3)

    with col_f1:
        uf_options = ['All'] + sorted(df['uf'].unique().tolist())
        uf_selected = st.selectbox("Selecionar UF", options=uf_options)

    with col_f2:
        regioes_options = ['All'] + sorted(df['regiao'].unique().tolist())
        regiao_selected = st.selectbox("Selecionar Região", options=regioes_options)

    with col_f3:
        mes_options = ['All'] + sorted(df['mes_texto'].unique().tolist())
        mes_selected = st.selectbox("Selecionar Mês", options=mes_options)

    with col_f4:
        modalidade_options = ['All'] + sorted(df['modalidade'].unique().tolist())
        modalidade_selected = st.selectbox("Selecionar Modalidade", options=modalidade_options)

    with col_f5:
        cliente_options = ['All'] + sorted(df['cliente'].unique().tolist())
        cliente_selected = st.selectbox("Selecionar Tipo de Cliente", options=cliente_options)

    with col_f6:
        porte_options = ['All'] + sorted(df['porte'].unique().tolist())
        porte_selected = st.selectbox("Selecionar Porte", options=porte_options)

    # Aplicar filtros
    df_filtrado = df

    if uf_selected != 'All':
        df_filtrado = df_filtrado[df_filtrado['uf'] == uf_selected]

    if regiao_selected != 'All':
        df_filtrado = df_filtrado[df_filtrado['regiao'] == regiao_selected]

    if mes_selected != 'All':
        df_filtrado = df_filtrado[df_filtrado['mes_texto'] == mes_selected]

    if modalidade_selected != 'All':
        df_filtrado = df_filtrado[df_filtrado['modalidade'] == modalidade_selected]

    if cliente_selected != 'All':
        df_filtrado = df_filtrado[df_filtrado['cliente'] == cliente_selected]

    if porte_selected != 'All':
        df_filtrado = df_filtrado[df_filtrado['porte'] == porte_selected]

    def formatar_valor(valor):
        if valor >= 1e12:
            return f" {valor/1e12:.2f}T"
        elif valor >= 1e9:
            return f" {valor/1e9:.2f}B"
        elif valor >= 1e6:
            return f" {valor/1e6:.2f}M"
        else:
            return f" {valor:,.2f}"

    # KPIs principais
    st.subheader("✨ Indicadores Principais")

    # Calcule os valores numéricos antes de formatá-los
    total_clientes = len(df_filtrado)
    total_inadimplentes = len(df_filtrado[df_filtrado['inadimplente'] == 1])
    percentual_inadimplencia = (total_inadimplentes / total_clientes) * 100
    total_carteira_ativa = df_filtrado['carteira_ativa'].sum()

    # Formate os valores para exibição
    total_clientes_formatado = formatar_valor(total_clientes)
    total_inadimplentes_formatado = formatar_valor(total_inadimplentes)
    total_carteira_ativa_formatado = formatar_valor(total_carteira_ativa)

    # Exibindo os KPIs em colunas
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Total de Clientes", value=total_clientes_formatado)

    with col2:
        st.metric(label="Total Inadimplentes", value=total_inadimplentes_formatado)

    with col3:
        st.metric(label="% Inadimplência", value=f"{percentual_inadimplencia:.2f}%")

    with col4:
        st.metric(label="Total Carteira Ativa (R$)", value=f"R$ {total_carteira_ativa_formatado}")


    # Gráfico 1: Mapa de Inadimplência por Estado
    st.subheader("🗺️ Mapa da Inadimplência por Estado")

    # Pega o GeoJSON dos estados
    url_geojson = 'https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson'
    geojson_estados = requests.get(url_geojson).json()

    # Agrupamento
    inadimplencia_por_estado = df_filtrado.groupby('uf').agg({
        'cliente': 'count',
        'inadimplente': 'sum'
    }).rename(columns={'cliente': 'quantidade_clientes'})

    inadimplencia_por_estado['taxa_inadimplencia'] = (
        inadimplencia_por_estado['inadimplente'] / inadimplencia_por_estado['quantidade_clientes']
    )

    inadimplencia_por_estado = inadimplencia_por_estado.reset_index()

    # 🔥 Criando o botão de seleção
    opcao = st.radio(
        "Escolha o tipo de dado para o mapa:",
        ("Taxa de Inadimplência (%)", "Valor Total de Inadimplentes"),
        horizontal=True
    )

    # 🔥 Definindo qual coluna usar
    if opcao == "Taxa de Inadimplência (%)":
        color_col = 'taxa_inadimplencia'
        color_label = 'Taxa de Inadimplência (%)'
        color_scale = 'pinkyl'
    else:
        color_col = 'inadimplente'
        color_label = 'Quantidade de Inadimplentes'
        color_scale = 'peach'

    # Gráfico
    fig1 = px.choropleth(
        inadimplencia_por_estado,
        geojson=geojson_estados,
        locations='uf',
        featureidkey="properties.sigla",
        color=color_col,
        color_continuous_scale=color_scale,
        labels={color_col: color_label},
        title=" "
    )

    fig1.update_geos(
        fitbounds="locations",
        visible=False
    )

    fig1.update_layout(
        title_x=0.5,
        margin={"r":0,"t":30,"l":0,"b":0},
        height=700  # Tamanho maior para melhorar visualização
    )

    st.plotly_chart(fig1, use_container_width=True)

    # Gráfico 2: Evolução Temporal da Inadimplência
    st.subheader("📅 Evolução Temporal da Inadimplência")

    # Contagem de inadimplentes por mês (supondo que 'mes' seja a coluna de tempo)
    inadimplentes_temporal = df_filtrado[df_filtrado['inadimplente'] == 1].groupby('mes_texto').size().reset_index(name='quantidade')

    # Definindo ordem dos meses
    ordem_meses = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 
                'Julho', 'Agosto', 'Setembro', 'Outubro']

    inadimplentes_temporal['mes_texto'] = pd.Categorical(
        inadimplentes_temporal['mes_texto'],
        categories=ordem_meses,
        ordered=True
    )
    inadimplentes_temporal = inadimplentes_temporal.sort_values('mes_texto')

    # Criar gráfico de linha
    fig2 = px.line(
        inadimplentes_temporal,
        x='mes_texto',
        y='quantidade',
        labels={'quantidade': 'Quantidade de Clientes Inadimplentes', 'mes_texto': 'Mês'},
        title=" ",
        color_discrete_sequence=['#FF5733'],  # Cor personalizada
        markers=True
    )

    # Exibir o gráfico
    st.plotly_chart(fig2, use_container_width=True)

    # Gráfico 3: Distribuição de Clientes por Status da Carteira
    st.subheader("📊 Distribuição de Clientes por Status da Carteira")
    # Criar gráfico de barras
    # Soma total por categoria
    resumo = pd.DataFrame({
        'Categoria': ['Carteira Ativa', 'Carteira Inadimplida Arrastada', 'Ativo Problemático'],
        'Quantidade': [
            df_filtrado['carteira_ativa'].sum(),
            df_filtrado['carteira_inadimplida_arrastada'].sum(),
            df_filtrado['ativo_problematico'].sum()
        ]
    })
    fig3 = px.bar(
        resumo,
        x='Categoria',
        y='Quantidade',
        color='Categoria',
        color_discrete_map={
            'Carteira Ativa': '#00bfae',
            'Carteira Inadimplida Arrastada': '#FF6347',
            'Ativo Problemático': '#f7a05c'
        },
        labels={'Quantidade': 'Número de Clientes', 'Categoria': 'Status da Carteira'},
        title=" "
    )

    # Exibir gráfico no Streamlit
    st.plotly_chart(fig3, use_container_width=True)

    # Gráfico 4: Inadimplência por Modalidade
    st.subheader("💼 Inadimplência por Modalidade de Crédito")

    inadimplencia_modalidade = df_filtrado.groupby('modalidade').agg({
        'cliente': 'count',
        'inadimplente': 'sum'
    }).rename(columns={'cliente': 'quantidade_clientes'})

    inadimplencia_modalidade['taxa_inadimplencia'] = (
        inadimplencia_modalidade['inadimplente'] / inadimplencia_modalidade['quantidade_clientes']
    )

    inadimplencia_modalidade = inadimplencia_modalidade.sort_values(by='taxa_inadimplencia', ascending=True)  # Para barra horizontal, crescente fica melhor

    # Gráfico
    fig4 = px.bar(
        inadimplencia_modalidade.reset_index(),
        x='taxa_inadimplencia',
        y='modalidade',
        color='taxa_inadimplencia',
        orientation='h', 
        color_continuous_scale='peach',
        title=" ",
        labels={
            'taxa_inadimplencia': 'Taxa de Inadimplência (%)',
            'modalidade': 'Modalidade'
        }
    )

    fig4.update_layout(
        title_x=0.5,
        xaxis_title='Taxa de Inadimplência (%)',
        yaxis_title='Modalidade',
        height=700,  # aumenta o tamanho pra caber os nomes
        margin={"r":30,"t":30,"l":30,"b":30}
    )

    st.plotly_chart(fig4, use_container_width=True)


    # Gráfico 5: Evolução no tempo
    st.subheader("📅 Evolução da Carteira ao Longo do Tempo")

    df_tempo = df_filtrado.groupby('data_base').sum(numeric_only=True).reset_index()
    fig5 = px.line(
        df_tempo,
        x='data_base',
        y=['carteira_ativa', 'carteira_inadimplida_arrastada', 'ativo_problematico'],
        markers=True,
        title=" "
    )
    st.plotly_chart(fig5, use_container_width=True)

    # Gráfico 6:  Valor Total a Vencer por Tempo
    st.subheader("📅  Valor Total a Vencer por Tempo")
    # Criação de um DataFrame para os valores por faixa de vencimento
    df_vencimentos = df_filtrado[['a_vencer_ate_90_dias', 
                                'a_vencer_de_91_ate_360_dias', 
                                'a_vencer_de_361_ate_1080_dias', 
                                'a_vencer_de_1081_ate_1800_dias', 
                                'a_vencer_de_1801_ate_5400_dias', 
                                'a_vencer_acima_de_5400_dias']].sum()

    # Criar gráfico de barras empilhadas com Plotly
    fig6 = go.Figure(data=[
        go.Bar(name='Até 90 dias', x=df_vencimentos.index, y=[df_vencimentos['a_vencer_ate_90_dias']]),
        go.Bar(name='91 a 360 dias', x=df_vencimentos.index, y=[df_vencimentos['a_vencer_de_91_ate_360_dias']]),
        go.Bar(name='361 a 1080 dias', x=df_vencimentos.index, y=[df_vencimentos['a_vencer_de_361_ate_1080_dias']]),
        go.Bar(name='1081 a 1800 dias', x=df_vencimentos.index, y=[df_vencimentos['a_vencer_de_1081_ate_1800_dias']]),
        go.Bar(name='1801 a 5400 dias', x=df_vencimentos.index, y=[df_vencimentos['a_vencer_de_1801_ate_5400_dias']]),
        go.Bar(name='Acima de 5400 dias', x=df_vencimentos.index, y=[df_vencimentos['a_vencer_acima_de_5400_dias']]),
    ])
    # Configurar layout do gráfico
    fig6.update_layout(
        barmode='stack',
        title=" ",
        xaxis_title="Faixa de Tempo",
        yaxis_title="Valor Total a Vencer (R$)",
        template="plotly_white",
    )

    # Exibir o gráfico no Streamlit
    st.plotly_chart(fig6, use_container_width=True)

    # Insight sobre a faixa de prazo que concentra mais dívida futura
    faixa_maxima = df_vencimentos.idxmax()
    valor_maximo = df_vencimentos.max()
    st.write(f"📊 A faixa de tempo que concentra mais dívida futura é: **{faixa_maxima}** com um total de R$ {valor_maximo:,.2f}")



elif pagina == "📊 Score de Inadimplência":
    st.title("Previsão de Inadimplência")
    st.markdown("Preencha os dados para prever a probabilidade de inadimplência:")

    # Formulário
    porte = st.selectbox('Porte', df['porte'].unique())
    ocupacao = st.selectbox('Ocupação', df['ocupacao'].unique())
    modalidade = st.selectbox('Modalidade', df['modalidade'].unique())
    total_a_vencer = st.number_input('Total a Vencer (R$)', min_value=0.0, step=100.0)
    vencido_acima_15 = st.number_input('Valor Vencido Acima de 15 Dias (R$)', min_value=0.0, step=100.0)

    if st.button("Calcular Probabilidade"):
        entrada = pd.DataFrame([{
            'porte': porte,
            'ocupacao': ocupacao,
            'modalidade': modalidade,
            'total_a_vencer': total_a_vencer,
            'vencido_acima_de_15_dias': vencido_acima_15
        }])

        for col in ['porte', 'ocupacao', 'modalidade']:
            le = LabelEncoder()
            le.fit(df[col])
            entrada[col] = le.transform(entrada[col])

        prob = modelo.predict_proba(entrada)[0][1]
        st.success(f"🔮 Probabilidade de Inadimplência: {prob:.2%}")
