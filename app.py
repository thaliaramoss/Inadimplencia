# app.py
from xgboost import XGBClassifier
import streamlit as st
import pandas as pd
import joblib

# Entradas do usuário
porte = st.selectbox('Porte', [
    'Mais de 3 a 5 salários mínimos',
    'Mais de 5 a 10 salários mínimos',
    'Mais de 1 a 2 salários mínimos',
    'Até 1 salário mínimo',
    'Mais de 10 a 20 salários mínimos',
    'Mais de 2 a 3 salários mínimos'
])

ocupacao = st.selectbox('Ocupação', [
    'Empregado de empresa privada',
    'Empregado de entidades sem fins lucrativos',
    'Empresário',
    'Servidor ou empregado público',
    'Aposentado/pensionista',
    'Autônomo'
])

modalidade = st.selectbox('Modalidade', [
    'Empréstimo sem consignação em folha',
    'Empréstimo com consignação em folha',
    'Veículos',
    'Cartão de crédito',
    'Habitacional'
])

origem = st.selectbox('Origem', [
    'Sem destinação específica',
    'Com destinação específica'
])

indexador = st.selectbox('Indexador', [
    'Prefixado',
    'Flutuantes',
    'Pós-fixado',
    'Outros indexadores',
    'Índices de preços',
    'TCR/TRFC'
])

carteira_ativa = st.number_input('Total a Vencer (R$)', min_value=0.0, step=100.0)
vencido_acima_15 = st.number_input('Valor Vencido Acima de 15 Dias (R$)', min_value=0.0, step=100.0)

# Prever inadimplência
if st.button("Calcular Probabilidade"):
    entrada = pd.DataFrame([{
        'ocupacao': ocupacao,
        'porte': porte,
        'modalidade': modalidade,
        'origem': origem,
        'indexador': indexador,
        'carteira_ativa': carteira_ativa,
        'vencido_acima_de_15_dias': vencido_acima_15
    }])

        for col in ['porte', 'ocupacao', 'modalidade']:
            le = LabelEncoder()
            le.fit(df[col])
            entrada[col] = le.transform(entrada[col])

        prob = modelo.predict_proba(entrada)[0][1]
        st.success(f"🔮 Probabilidade de Inadimplência: {prob:.2%}")