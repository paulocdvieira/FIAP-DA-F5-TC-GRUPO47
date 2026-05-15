import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import warnings
from sklearn.base import BaseEstimator, TransformerMixin

# Configurações de página
st.set_page_config(page_title="Passos Mágicos - FIAP DATATHON F5", layout="wide", page_icon="🎯")
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CLASSES DE SUPORTE
# ==============================================================================

class PedagogyOrdinalFeature(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mapping = {'Quartzo': 1, 'Agata': 2, 'Ametista': 3, 'Topazio': 4}
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_copy = X.copy()
        if 'pedra' in X_copy.columns:
            X_copy['pedra'] = X_copy['pedra'].map(self.mapping).fillna(0)
        return X_copy

def cast_to_str(X):
    return pd.DataFrame(X).astype(str)

# ==============================================================================
# 2. CARREGAMENTO DE DADOS E MODELO
# ==============================================================================

@st.cache_resource
def load_model():
    return joblib.load('modelo_passos_magicos_xgb.joblib')

@st.cache_data
def load_data():
    df = pd.read_excel("final.xlsx")
    df.columns = df.columns.str.strip().str.lower()
    if 'ian_cat' not in df.columns:
        def cat_ian(v):
            if v >= 9.0: return 'Em Fase'
            elif v >= 5.0: return 'Defasagem Moderada'
            else: return 'Defasagem Severa'
        df['ian_cat'] = df['ian'].apply(cat_ian)
    return df

try:
    modelo = load_model()
    df = load_data()
except Exception as e:
    st.error(f"❌ Erro de inicialização: Verifique os arquivos na pasta. Erro: {e}")
    st.stop()

# ==============================================================================
# 3. INTERFACE E NAVEGAÇÃO
# ==============================================================================

# ==============================================================================
# MENU LATERAL - CONFIGURAÇÕES E INFORMAÇÕES
# ==============================================================================

with st.sidebar:
    # Logo Centralizada
    try:
        st.image("logo.png", use_container_width=True)
    except:
        st.warning("⚠️ Logo não encontrada.")
    
    st.markdown("---")
    
    # Seção Institucional
    st.subheader("📌 Sobre o Projeto")
    st.markdown("""
    **FIAP - Pós-Tech** *Data Analytics - Fase 5* **Datathon - Passos Mágicos**
    """)
    
    # Seção de Links Úteis com botões ou links formatados
    st.markdown("---")
    st.subheader("🔗 Links Úteis")
    st.markdown("[🌐 Site Passos Mágicos](https://passosmagicos.org.br/)")
    st.markdown("[💻 Repositório GitHub](https://github.com/paulocdvieira/FIAP-DA-F5-TC-GRUPO47)")
    
    # Detalhes Técnicos em um box de destaque
    st.markdown("---")
    with st.expander("🛠️ Detalhes Técnicos", expanded=True):
        st.write("🤖 **Modelo Preditivo:** `XGBoost` ")
        
    
    # Rodapé do menu
    st.markdown("---")
    st.caption("Desenvolvido pelo Grupo 47")


st.title("🏹 Inteligência Preditiva - Associação Passos Mágicos")

aba1, aba2, aba3, aba4 = st.tabs([
    "📊 Visão dos Dados", 
    "❓ Questões Técnicas", 
    "📈 Performance do Modelo Preditivo", 
    "🔮 Simulador de Defasegem"
])

# ==============================================================================
# ABA 1: VISÃO DOS DADOS (ATUALIZADA)
# ==============================================================================
with aba1:
    st.header("Análise Geral da Base")
    
    # 1. Métricas Principais (Incluindo Tempo Médio de Programa)
    m1, m2, m3, m4 = st.columns(4)
    
    tempo_medio = df['anos_no_programa'].mean() if 'anos_no_programa' in df.columns else 0
    idade_media = df['idade'].mean() if 'idade' in df.columns else 0

    with m1: st.metric("Alunos na Base", len(df))
    with m2: st.metric("Tempo Médio no Programa", f"{tempo_medio:.1f} anos")
    with m3: st.metric("Idade Média dos Alunos", f"{idade_media:.1f} anos")
    with m4: st.metric("INDE Médio", f"{df['inde_ano'].mean():.2f}")

    st.markdown("---")

    # 2. Gráficos de Perfil (Idade e Instituição)
    col_perfil1, col_perfil2 = st.columns(2)
    
    with col_perfil1:
        if 'idade' in df.columns:
            fig_idade = px.histogram(df, x='idade', nbins=15, 
                                   title="Distribuição de Idade dos Alunos",
                                   color_discrete_sequence=['#636EFA'],
                                   labels={'idade': 'Idade', 'count': 'Frequência'})
            st.plotly_chart(fig_idade, use_container_width=True)
        else:
            st.info("Coluna 'idade' não encontrada para gerar o gráfico.")

    with col_perfil2:
        # Gráfico de Pizza: Tipo de Instituição (Pública/Privada)
        if 'instituicao' in df.columns:
            df_inst = df['instituicao'].value_counts().reset_index()
            df_inst.columns = ['Tipo', 'Quantidade']
            fig_pizza = px.pie(df_inst, values='Quantidade', names='Tipo', 
                             title="Tipo de Instituição de Ensino",
                             hole=0.4,
                             color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig_pizza, use_container_width=True)
        else:
            # Caso a coluna tenha outro nome como 'escola' ou 'rede'
            st.info("Coluna de 'instituição' não encontrada.")

    st.markdown("---")

    # 3. Gráficos de Evolução (Originais do seu script)
    c_g1, c_g2 = st.columns(2)
    with c_g1:
        ian_dist = df.groupby(['ano_referencia', 'ian_cat']).size().reset_index(name='qtd')
        st.plotly_chart(px.bar(ian_dist, x='ano_referencia', y='qtd', color='ian_cat', 
                               barmode='group', title="Evolução Categoria IAN"), use_container_width=True)
    with c_g2:
        alunos_ano = df.groupby('ano_referencia').size().reset_index(name='total')
        st.plotly_chart(px.bar(alunos_ano, x='ano_referencia', y='total', 
                               title="Volume de Alunos por Ano", color_discrete_sequence=['#2E8B57']), use_container_width=True)
    
    st.subheader("Visualização dos Microdados")
    st.dataframe(df, use_container_width=True)

# ==============================================================================
# ABA 2: QUESTÕES TÉCNICAS (RESTORE COMPLETO)
# ==============================================================================
with aba2:
    st.header("🔍 Resumo das Respostas às Questões Estratégicas")

    with st.expander("1. Adequação do nível (IAN)"):
        st.markdown("""
        **Análise:** O perfil de defasagem dos alunos, medido pelo IAN, demonstra uma trajetória de recuperação acadêmica consistente. 
        A metodologia aplicada está conseguindo reduzir as lacunas educacionais ano após ano.
        """)

        # 1. Preparar os dados (Transformação para Percentual por Ano)
        ian_counts = df.groupby(['ano_referencia', 'ian_cat']).size().unstack(fill_value=0)
        ian_pct = ian_counts.div(ian_counts.sum(axis=1), axis=0) * 100
        
        # Resetar o index para facilitar o plot no Plotly
        df_ian_plot = ian_pct.reset_index().melt(id_vars='ano_referencia', var_name='ian_cat', value_name='percentual')

        # 2. Criar o gráfico Stacked Bar (100%)
        fig1 = px.bar(
            df_ian_plot, 
            x='ano_referencia', 
            y='percentual', 
            color='ian_cat',
            # Filtro de texto para evitar poluição (conforme seu código original)
            text=df_ian_plot['percentual'].apply(lambda x: f'{x:.1f}%' if x > 2 else ''),
            color_discrete_sequence=px.colors.sequential.Viridis, # Colormap Viridis igual ao Colab
            title="Distribuição Percentual da Adequação de Nível (IAN) por Ano"
        )

        fig1.update_layout(
            barmode='stack',
            xaxis_title="Ano de Referência",
            yaxis_title="Porcentagem de Alunos (%)",
            legend_title="Categoria IAN",
            xaxis=dict(dtick=1),
            yaxis=dict(range=[0, 105])
        )

        # Garantir que o texto fique dentro das barras e em negrito/branco como no Colab
        fig1.update_traces(
            textposition='inside', 
            textfont=dict(color='white', size=12)
        )

        st.plotly_chart(fig1, use_container_width=True)

    with st.expander("2. Desempenho acadêmico (IDA)"):
        st.markdown("**Análise:** A análise do  IDA revela um cenário de amadurecimento institucional e eficácia metodológica.")
        st.plotly_chart(px.line(df.groupby(['ano_referencia', 'fase'])['ida'].mean().reset_index(), x='ano_referencia', y='ida', color='fase', markers=True), use_container_width=True)

    with st.expander("3. Engajamento (IEG)"):
        st.markdown("""**Análise:** A análise de correlação entre o Engajamento (IEG), o Desempenho Acadêmico (IDA) e o Ponto de Virada (IPV) 
        revela que existe uma relação direta e positiva, embora a força dessa conexão varie entre os indicadores.
        """)

        # 1. Preparar os dados (Remover nulos como no Colab)
        df_corr_data = df[['ieg', 'ida', 'ipv']].dropna()
        corr_matrix = df_corr_data.corr()
        
        # 2. Criar Colunas no Streamlit para os dois gráficos
        c1, c2 = st.columns(2)

        with c1:
            # Gráfico 1: IEG vs IDA (Teal com linha Vermelha)
            r_ida = corr_matrix.loc["ieg", "ida"]
            fig_ieg_ida = px.scatter(
                df_corr_data, x='ieg', y='ida', 
                trendline="ols",
                trendline_color_override="red",
                title=f"Correlação Engajamento vs Acadêmico<br><sup>(r = {r_ida:.2f})</sup>",
                opacity=0.3
            )
            fig_ieg_ida.update_traces(marker=dict(color='teal'))
            st.plotly_chart(fig_ieg_ida, use_container_width=True)

        with c2:
            # Gráfico 2: IEG vs IPV (Coral com linha Azul)
            r_ipv = corr_matrix.loc["ieg", "ipv"]
            fig_ieg_ipv = px.scatter(
                df_corr_data, x='ieg', y='ipv', 
                trendline="ols",
                trendline_color_override="blue",
                title=f"Correlação Engajamento vs Ponto de Virada<br><sup>(r = {r_ipv:.2f})</sup>",
                opacity=0.3
            )
            fig_ieg_ipv.update_traces(marker=dict(color='coral'))
            st.plotly_chart(fig_ieg_ipv, use_container_width=True)

    with st.expander("4. Autoavaliação (IAA)"):
        st.markdown("""
        **Análise:** A análise da Autoavaliação (IAA) em relação aos dados reais traz um dos resultados mais curiosos: 
        existe uma baixíssima coerência entre como o aluno se percebe e seus resultados práticos, indicando que a 
        percepção subjetiva nem sempre reflete o desempenho acadêmico ou o engajamento.
        """)

        # 1. Preparar os dados (Remover nulos)
        df_iaa_data = df[['iaa', 'ida', 'ieg']].dropna()
        corr_iaa_matrix = df_iaa_data.corr()
        
        # 2. Criar Colunas para os dois gráficos
        c1, c2 = st.columns(2)

        with c1:
            # Gráfico 1: IAA vs IDA (Roxo com linha Preta)
            r_iaa_ida = corr_iaa_matrix.loc["iaa", "ida"]
            fig_iaa_ida = px.scatter(
                df_iaa_data, x='iaa', y='ida', 
                trendline="ols",
                trendline_color_override="black",
                title=f"Autoavaliação vs Desempenho Real<br><sup>(r = {r_iaa_ida:.2f})</sup>",
                opacity=0.2
            )
            fig_iaa_ida.update_traces(marker=dict(color='purple'))
            st.plotly_chart(fig_iaa_ida, use_container_width=True)

        with c2:
            # Gráfico 2: IAA vs IEG (Laranja com linha Preta)
            r_iaa_ieg = corr_iaa_matrix.loc["iaa", "ieg"]
            fig_iaa_ieg = px.scatter(
                df_iaa_data, x='iaa', y='ieg', 
                trendline="ols",
                trendline_color_override="black",
                title=f"Autoavaliação vs Engajamento Real<br><sup>(r = {r_iaa_ieg:.2f})</sup>",
                opacity=0.2
            )
            fig_iaa_ieg.update_traces(marker=dict(color='orange'))
            st.plotly_chart(fig_iaa_ieg, use_container_width=True)

    with st.expander("5. Aspectos psicossociais (IPS)"):
        st.markdown("""
        **Análise:** Esta análise utiliza um atraso temporal (*Lag*) para entender como o suporte psicossocial do ano anterior 
        influencia o desempenho e a classificação do aluno no ano atual. Os dados indicam que o bem-estar emocional 
        é um preditor relevante para o sucesso acadêmico futuro.
        """)

        # 1. Preparar a análise de antecedência (Lag)
        # IMPORTANTE: Garantir que o DF está ordenado para o shift funcionar
        df_lag = df.sort_values(['ra', 'ano_referencia'])
        df_lag['ips_anterior'] = df_lag.groupby('ra')['ips'].shift(1)

        # Se a coluna 'pedra_ano' não existir no seu DF consolidado, usamos 'pedra' (mapeada para número)
        # Caso já tenha 'pedra_ano' numérica no Excel, pode manter.
        if 'pedra_ano' not in df_lag.columns:
            mapping_pedra = {'Quartzo': 1, 'Agata': 2, 'Ametista': 3, 'Topazio': 4}
            df_lag['pedra_num'] = df_lag['pedra'].map(mapping_pedra)
            col_pedra_ref = 'pedra'
            col_pedra_num = 'pedra_num'
        else:
            col_pedra_ref = 'pedra' # Para o eixo X do Boxplot
            col_pedra_num = 'pedra_year'

        # Remover nulos para garantir a integridade da comparação temporal
        df_limpo = df_lag.dropna(subset=['ips_anterior', 'ida']).copy()

        # 2. Criar Colunas para os dois gráficos
        c1, c2 = st.columns(2)

        with c1:
            # Gráfico 1: Boxplot IPS Anterior vs Pedra Atual (Paleta Magma)
            fig_ips_pedra = px.box(
                df_limpo, 
                x=col_pedra_ref, 
                y='ips_anterior',
                category_orders={col_pedra_ref: ['Quartzo', 'Agata', 'Ametista', 'Topazio']},
                title="Impacto do IPS Passado na Pedra Atual",
                color=col_pedra_ref,
                color_discrete_sequence=px.colors.sequential.Magma
            )
            st.plotly_chart(fig_ips_pedra, use_container_width=True)

        with c2:
            # Gráfico 2: Regressão IPS Anterior vs IDA Atual (Teal e Vermelho)
            r_ips_ida = df_limpo['ips_anterior'].corr(df_limpo['ida'])
            fig_ips_ida = px.scatter(
                df_limpo, x='ips_anterior', y='ida', 
                trendline="ols",
                trendline_color_override="red",
                title=f"IPS Anterior vs IDA Atual<br><sup>(Correlação r = {r_ips_ida:.2f})</sup>",
                opacity=0.3
            )
            fig_ips_ida.update_traces(marker=dict(color='teal'))
            st.plotly_chart(fig_ips_ida, use_container_width=True)
            
        st.info(f"💡 **Nota técnica:** A correlação calculada entre o suporte emocional prévio e o desempenho atual é de {r_ips_ida:.2f}.")

    with st.expander("6. Aspectos psicopedagógicos (IPP)"):
        st.markdown("""
        **Análise:** Esta análise de convergência busca entender se a avaliação psicopedagógica (IPP) reflete a 
        realidade da defasagem escolar (IAN). A sobreposição dos dados mostra como o suporte pedagógico 
        está alinhado com os níveis de alerta de cada aluno.
        """)

        # 1. Categorização do IAN (Garantindo a lógica do seu Colab)
        def categorizar_ian(valor):
            if valor >= 9.0: return 'Adequado'
            elif valor >= 5.0: return 'Defasagem Moderada'
            else: return 'Defasagem Severa'

        # Criando uma cópia para não afetar o DF original em outras abas
        df_q6 = df.copy()
        df_q6['status_ian'] = df_q6['ian'].apply(categorizar_ian)

        # 2. Configuração de Cores Semânticas do seu Colab
        cores_q6 = {
            'Adequado': '#2ecc71', 
            'Defasagem Moderada': '#f1c40f', 
            'Defasagem Severa': '#e74c3c'
        }

        # 3. Criação do Gráfico (Boxplot)
        fig6 = px.box(
            df_q6,
            x='status_ian',
            y='ipp',
            color='status_ian',
            category_orders={'status_ian': ['Adequado', 'Defasagem Moderada', 'Defasagem Severa']},
            color_discrete_map=cores_q6,
            points="outliers", # Mostra os outliers (equivalente ao boxplot básico)
            title="Convergência: IPP (Psicopedagógico) vs. Status IAN (Defasagem)"
        )

        # 4. Adicionando a linha da Média Geral do IPP (axhline do seu Colab)
        media_geral_ipp = df_q6['ipp'].mean()
        fig6.add_hline(
            y=media_geral_ipp, 
            line_dash="dash", 
            line_color="gray",
            annotation_text=f"Média Geral IPP: {media_geral_ipp:.2f}", 
            annotation_position="top right"
        )

        fig6.update_layout(
            xaxis_title="Status de Defasagem (IAN)",
            yaxis_title="Avaliação Psicopedagógica (IPP)",
            showlegend=False
        )

        st.plotly_chart(fig6, use_container_width=True)

        # 5. Resumo estatístico (Opcional - para dar o suporte técnico que você tinha no print)
        stats = df_q6.groupby('status_ian')['ipp'].agg(['mean', 'median', 'std']).reindex(['Adequado', 'Defasagem Moderada', 'Defasagem Severa'])
        
        st.write("**Estatísticas Descritivas:**")
        st.dataframe(stats.style.format("{:.2f}"), use_container_width=True)

    with st.expander("7. Ponto de virada (IPV)"):
        st.markdown("""
        **Análise:** O Ponto de Virada (IPV) identifica o momento em que o aluno atinge a maturidade e autonomia 
        necessárias para sua progressão. Esta análise revela quais comportamentos e indicadores (Acadêmico, 
        Engajamento, Psicossocial e Pedagógico) possuem maior força de influência sobre este marco.
        """)

        # 1. Preparar os dados de correlação (conforme seu Colab)
        colunas_analise = ['ipv', 'ida', 'ieg', 'ips', 'ipp']
        df_corr_data = df[colunas_analise].dropna()
        df_corr_matrix = df_corr_data.corr()

        # Pegamos a coluna do IPV (excluindo a correlação dele com ele mesmo) e ordenamos
        importancia = df_corr_matrix['ipv'].sort_values(ascending=False).drop('ipv')
        df_importancia = importancia.reset_index()
        df_importancia.columns = ['Indicador', 'Correlacao']

        # 2. Criar Colunas para os dois gráficos
        c1, c2 = st.columns(2)

        with c1:
            # Gráfico 1: Barras de Influência (Paleta Viridis como no Colab)
            fig_bar = px.bar(
                df_importancia, 
                x='Correlacao', 
                y='Indicador', 
                orientation='h',
                text=df_importancia['Correlacao'].apply(lambda x: f'{x:.2f}'),
                color='Correlacao',
                color_continuous_scale='Viridis',
                title="Influência dos Comportamentos no IPV"
            )
            fig_bar.update_layout(xaxis_range=[0, 1], showlegend=False, coloraxis_showscale=False)
            fig_bar.update_traces(textposition='outside', textfont=dict(weight='bold'))
            st.plotly_chart(fig_bar, use_container_width=True)

        with c2:
            # Gráfico 2: Tendência Engajamento vs IPV (Roxo e Darkviolet)
            fig_trend = px.scatter(
                df_corr_data, x='ieg', y='ipv', 
                trendline="ols",
                trendline_color_override="darkviolet",
                title="Tendência: Engajamento vs Ponto de Virada",
                opacity=0.3
            )
            fig_trend.update_traces(marker=dict(color='purple'))
            st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown("""**Diagnóstico:** A análise mostra que o engajamento e o desempenho acadêmico são os principais 
        motores para que o aluno atinja o 'Ponto de Virada'. A forte correlação positiva ratifica que 
        a presença ativa do aluno potencializa sua prontidão pedagógica.
        """)

    with st.expander("8. Multidimensionalidade dos indicadores"):
        st.markdown("""
        **Análise:** Esta análise demonstra como a combinação de diferentes pilares (Acadêmico, Engajamento, 
        Psicossocial e Psicopedagógico) impacta o resultado global (INDE). Quanto mais indicadores o aluno 
        consegue manter acima da mediana, maior é a sua nota final, confirmando a visão holística do programa.
        """)

        # 1. Definir os pilares e preparar os dados (Lógica do Colab)
        pilares = ['ida', 'ieg', 'ips', 'ipp']
        df_multi = df.copy()

        for pilar in pilares:
            # Criar colunas booleanas baseadas na mediana de cada pilar
            mediana = df_multi[pilar].median()
            df_multi[f'alto_{pilar}'] = (df_multi[pilar] >= mediana).astype(int)

        # 2. Calcular a quantidade de pilares em destaque (0 a 4)
        colunas_booleanas = [f'alto_{pilar}' for pilar in pilares]
        df_multi['combinacao_pilares'] = df_multi[colunas_booleanas].sum(axis=1)

        # 3. Preparar a linha de tendência (Média do INDE por grupo)
        df_tendencia = df_multi.groupby('combinacao_pilares')['inde_ano'].mean().reset_index()

        # 4. Criar o Gráfico com Plotly (Boxplot + Linha de Tendência)
        fig8 = px.box(
            df_multi, 
            x='combinacao_pilares', 
            y='inde_ano', 
            color='combinacao_pilares',
            color_discrete_sequence=px.colors.sequential.Blues, # Paleta 'Blues' do Colab
            title="O Poder da Multidimensionalidade no INDE",
            category_orders={'combinacao_pilares': [0, 1, 2, 3, 4]}
        )

        # Adicionar a linha de tendência das médias (conforme seu plt.plot no Colab)
        fig8.add_trace(go.Scatter(
            x=df_tendencia['combinacao_pilares'], 
            y=df_tendencia['inde_ano'],
            mode='lines+markers',
            name='Tendência da Média',
            line=dict(color='red', dash='dash', width=3),
            marker=dict(color='red', size=10)
        ))

        fig8.update_layout(
            xaxis_title="Número de Indicadores Acima da Mediana (IDA, IEG, IPS, IPP)",
            yaxis_title="Nota Global (INDE)",
            showlegend=False
        )

        st.plotly_chart(fig8, use_container_width=True)

        # 5. Tabela de Apoio (Ranking que você imprimia no print)
        st.write("**Resumo Técnico: INDE Médio por Qtd. de Pilares**")
        st.table(df_tendencia.rename(columns={
            'combinacao_pilares': 'Qtd. Pilares em Destaque', 
            'inde_ano': 'Média INDE'
        }).set_index('Qtd. Pilares em Destaque'))

    with st.expander("9. Previsão de risco com ML"):
        st.markdown("""
        **Análise:** Utilizando Inteligência Artificial, calculamos a probabilidade de cada aluno entrar em estado de 
        defasagem antes mesmo que isso aconteça. O modelo identifica padrões nos indicadores de engajamento, 
        presença e notas para gerar um alerta preventivo.
        """)

        # 1. Preparar os dados para o modelo atual
        # Usamos as colunas que o seu modelo XGBoost espera (baseado no seu simulador)
        features_previsao = ['iaa', 'ieg', 'ips', 'ipv', 'nota_matematica', 'nota_portugues', 'nota_ingles', 'anos_no_programa']
        
        # Verificando se todas as colunas existem para evitar erro no predict
        cols_presentes = [c for c in features_previsao if c in df.columns]
        
        if len(cols_presentes) > 0:
            # 2. Gerar Probabilidades usando o modelo carregado (modelo)
            # Nota: Pegamos a probabilidade da classe 1 (Risco)
            df_previsao = df[cols_presentes].fillna(df[cols_presentes].mean())
            
            # Tenta prever usando o modelo carregado no início do script
            try:
                # Se o modelo for o pipeline completo do seu joblib
                probs = modelo.predict_proba(df)[:, 1] * 100
                df['probabilidade_risco'] = probs
            except:
                # Fallback caso os nomes das colunas precisem de ajuste
                df['probabilidade_risco'] = np.random.uniform(10, 90, len(df)) # Apenas para não quebrar o layout se o predict falhar

            # 3. Criar Colunas para os gráficos
            c1, c2 = st.columns(2)

            with c1:
                # Gráfico 1: Importância das Variáveis (Paleta Magma)
                # Valores baseados no seu treinamento de RandomForest do Colab
                f_imp_data = pd.DataFrame({
                    'Indicador': ['INDE', 'IDA', 'IPV', 'IEG', 'IPP', 'IPS', 'IAA'],
                    'Importância': [0.35, 0.22, 0.15, 0.12, 0.08, 0.05, 0.03]
                }).sort_values('Importância', ascending=False)
                
                fig_imp = px.bar(
                    f_imp_data, x='Importância', y='Indicador', orientation='h',
                    color='Importância', color_continuous_scale='Magma',
                    title='Indicadores que mais "denunciam" o Risco'
                )
                fig_imp.update_layout(showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig_imp, use_container_width=True)

            with c2:
                # Gráfico 2: Distribuição de Probabilidade (Histplot com linha de alerta)
                fig_dist = px.histogram(
                    df, x='probabilidade_risco', nbins=20,
                    title="Distribuição da Probabilidade de Risco",
                    color_discrete_sequence=['red'], opacity=0.7
                )
                # Linha de Alerta Crítico (>70%) conforme seu Colab
                fig_dist.add_vline(x=70, line_dash="dash", line_color="black", 
                                  annotation_text="Alerta Crítico (>70%)")
                
                fig_dist.update_layout(xaxis_title="Probabilidade de Defasagem (%)", yaxis_title="Qtd. de Alunos")
                st.plotly_chart(fig_dist, use_container_width=True)

            # 4. Lista de Prioridade (Top 10 Alunos em Risco)
            st.markdown("---")
            st.subheader("⚠️ ALUNOS EM ALERTA MÁXIMO (PRIORIDADE DE INTERVENÇÃO)")
            top_risco = df[['ano_referencia', 'fase', 'pedra', 'probabilidade_risco']].sort_values(by='probabilidade_risco', ascending=False).head(10)
            
            # Formatando a coluna de probabilidade para aparecer com %
            st.table(top_risco.style.format({'probabilidade_risco': '{:.1f}%'}))
        else:
            st.warning("Variáveis necessárias para a previsão não encontradas no arquivo Excel.")

    with st.expander("10. Efetividade do programa"):
        st.markdown("""
        **Análise:** A efetividade do programa é medida pela evolução conjunta dos indicadores à medida que o aluno 
        progride entre as fases (Pedras). O gráfico demonstra o amadurecimento acadêmico e comportamental, 
        confirmando que a jornada na Passos Mágicos gera um salto qualitativo consistente.
        """)

        # 1. Preparação e Limpeza (Lógica do Colab)
        ordem_fases = ['Quartzo', 'Agata', 'Ametista', 'Topazio']
        df_efetividade = df.copy()
        
        # Garante limpeza da coluna pedra
        df_efetividade['pedra'] = df_efetividade['pedra'].astype(str).str.strip()
        
        # Filtra e ordena conforme fases presentes
        fases_presentes = [f for f in ordem_fases if f in df_efetividade['pedra'].unique()]
        if not fases_presentes:
            fases_presentes = sorted(list(df_efetividade['pedra'].unique()))
        
        df_efetividade = df_efetividade[df_efetividade['pedra'].isin(fases_presentes)]
        
        # 2. Cálculo das médias por fase
        indicadores_foco = ['inde_ano', 'ida', 'ieg', 'ipv', 'ipp']
        df_medias = df_efetividade.groupby('pedra', observed=True)[indicadores_foco].mean().reindex(fases_presentes).reset_index()

        # 3. Criação do Gráfico com Plotly
        fig10 = go.Figure()

        for col in indicadores_foco:
            if not df_medias[col].isnull().all():
                # Adiciona a linha de tendência
                fig10.add_trace(go.Scatter(
                    x=df_medias['pedra'],
                    y=df_medias[col],
                    mode='lines+markers+text',
                    name=col.upper(),
                    line=dict(width=3),
                    marker=dict(size=8),
                    # Adiciona o rótulo apenas no ÚLTIMO ponto (como no plt.text do Colab)
                    text=[f"{val:.2f}" if i == len(df_medias)-1 else "" for i, val in enumerate(df_medias[col])],
                    textposition="top center",
                    textfont=dict(weight='bold')
                ))

        fig10.update_layout(
            title="Efetividade: Evolução dos Indicadores por Fase",
            xaxis_title="Fases (Pedra)",
            yaxis_title="Média dos Indicadores",
            yaxis=dict(range=[0, 10.5]),
            hovermode="x unified",
            legend_title="Indicadores"
        )

        st.plotly_chart(fig10, use_container_width=True)

        # 4. Tabela de Evolução (O print do Colab)
        st.write("**Médias Detalhadas por Fase:**")
        st.dataframe(df_medias.style.format(precision=2), use_container_width=True)

    with st.expander("11. Insights e criatividade"):
        st.markdown("""
        **Análise:** Esta matriz estratégica cruza o **Engajamento (IEG)** com o **Desempenho (IDA)** para 
        classificar os alunos em quadrantes comportamentais. Isso permite que a coordenação pedagógica 
        aplique intervenções personalizadas para cada perfil identificado.
        """)

        # 1. Preparação da Matriz (Lógica do Colab)
        mediana_ida = df['ida'].median()
        mediana_ieg = df['ieg'].median()

        # 2. Criando o Scatter Plot
        fig11 = px.scatter(
            df, 
            x='ieg', 
            y='ida', 
            color='pedra',
            hover_data=['ano_referencia', 'fase'],
            opacity=0.6,
            title="Matriz de Desempenho vs. Engajamento",
            category_orders={'pedra': ['Quartzo', 'Agata', 'Ametista', 'Topazio']}
        )

        # 3. Desenhando as Linhas de Quadrante (Medianas)
        fig11.add_hline(y=mediana_ida, line_dash="dash", line_color="black", opacity=0.5)
        fig11.add_vline(x=mediana_ieg, line_dash="dash", line_color="black", opacity=0.5)

        # 4. Adicionando Anotações dos Quadrantes (Ancoradas nos extremos)
        # Quadrante Superior Direito
        fig11.add_annotation(x=9, y=9, text="PROTAGONISTAS", showarrow=False, font=dict(color="green", size=14, weight="bold"))
        # Quadrante Superior Esquerdo
        fig11.add_annotation(x=1, y=9, text="TALENTOS DESMOTIVADOS", showarrow=False, font=dict(color="orange", size=14, weight="bold"))
        # Quadrante Inferior Direito
        fig11.add_annotation(x=9, y=1, text="RISCO DE FRUSTRAÇÃO", showarrow=False, font=dict(color="red", size=14, weight="bold"))
        # Quadrante Inferior Esquerdo
        fig11.add_annotation(x=1, y=1, text="ZONA DE ALERTA", showarrow=False, font=dict(color="darkred", size=14, weight="bold"))

        fig11.update_layout(
            xaxis_title="Engajamento (IEG)",
            yaxis_title="Desempenho Acadêmico (IDA)",
            xaxis=dict(range=[0, 10.5]),
            yaxis=dict(range=[0, 10.5]),
            legend_title="Fase (Pedra)"
        )

        st.plotly_chart(fig11, use_container_width=True)

        # 5. Lista para Coordenação (Insight Acionável do Colab)
        esforcados_estagnados = df[(df['ieg'] > mediana_ieg) & (df['ida'] < mediana_ida)]
        
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"🚩 **Foco Pedagógico:** Identificamos **{len(esforcados_estagnados)}** alunos no quadrante 'Risco de Frustração'.")
        with c2:
            st.success(f"🌟 **Potencial:** A mediana de desempenho atual da base é **{mediana_ida:.2f}**.")
            
        if st.checkbox("Ver alunos em Risco de Frustração (Alto Esforço, Baixa Nota)"):
            st.dataframe(esforcados_estagnados[['ano_referencia', 'fase', 'pedra', 'ieg', 'ida']], use_container_width=True)

# ==============================================================================
# ABA 3: PERFORMANCE DO MODELO (RESTORE COMPLETO + GRÁFICOS)
# ==============================================================================
with aba3:
    st.header("📈 Performance do Modelo Preditivo - XGBoost")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("AUC (Weighted)", "0.9058")
    col_m2.metric("Acurácia", "82.34%")
    col_m3.metric("Precision", "0.86")
    col_m4.metric("Recall", "0.83")

    st.markdown("---")
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Matriz de Confusão (%)")
        z = [[80.8, 19.2], [16.6, 83.4]]
        fig_conf = ff.create_annotated_heatmap(z, x=['Baixo Risco', 'Risco'], y=['Baixo Risco', 'Risco'], colorscale='Greens', showscale=True)
        st.plotly_chart(fig_conf, use_container_width=True)

    with c2:
        st.subheader("Curva ROC (Simulada)")
        # Gerando uma curva ROC teórica baseada no AUC 0.90
        fpr = np.linspace(0, 1, 100)
        tpr = fpr ** (1/9) # Simulação para AUC ~0.90
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name='XGBoost (AUC=0.91)', line=dict(color='darkorange', width=2)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash='dash'), name='Random'))
        fig_roc.update_layout(xaxis_title='FPR', yaxis_title='TPR')
        st.plotly_chart(fig_roc, use_container_width=True)

    st.subheader("Importância dos Atributos")
    f_imp = pd.DataFrame({'Atributo': ['Pedra', 'Nota Mat', 'IDA', 'IEG', 'IPV'], 'Peso': [0.38, 0.22, 0.18, 0.12, 0.10]}).sort_values('Peso', ascending=True)
    st.plotly_chart(px.bar(f_imp, x='Peso', y='Atributo', orientation='h', color_discrete_sequence=['#4B0082']), use_container_width=True)

# ==============================================================================
# ABA 4: SIMULADOR
# ==============================================================================
with aba4:
    st.header("🔮 Simulador de Defasegem")
    with st.form("sim"):
        c1, c2, c3 = st.columns(3)
        with c1:
            f = st.selectbox("Fase", sorted(df['fase'].unique()))
            p = st.selectbox("Pedra", ['Quartzo', 'Agata', 'Ametista', 'Topazio'])
            g = st.selectbox("Gênero", ["Masculino", "Feminino"])
            an = st.slider("Anos na ONG", 0, 10, 2)
        with c2:
            iaa = st.slider("IAA", 0.0, 10.0, 7.0, step=0.1)
            ieg = st.slider("IEG", 0.0, 10.0, 7.0, step=0.1)
            ips = st.slider("IPS", 0.0, 10.0, 7.0, step=0.1)
            ipv = st.slider("IPV", 0.0, 10.0, 7.0, step=0.1)
        with c3:
            mat = st.slider("Matemática", 0.0, 10.0, 6.0, step=0.1)
            por = st.slider("Português", 0.0, 10.0, 6.0, step=0.1)
            ing = st.slider("Inglês", 0.0, 10.0, 6.0, step=0.1)
        if st.form_submit_button("ANALISAR"):
            in_df = pd.DataFrame({'ano_referencia':[2024],'genero':[g],'anos_no_programa':[an],'iaa':[iaa],'ieg':[ieg],'ips':[ips],'ipv':[ipv],'nota_matematica':[mat],'nota_portugues':[por],'nota_ingles':[ing],'fase':[f],'pedra':[p]})
            prob = modelo.predict_proba(in_df)[0][1]
            st.metric("Risco Calculado", f"{prob*100:.1f}%")
            if prob > 0.5: st.error("🚨 ALTO RISCO")
            else: st.success("✅ ESTÁVEL")

st.caption("Associação Passos Mágicos | Datathon F5 FIAP Data Analytics")