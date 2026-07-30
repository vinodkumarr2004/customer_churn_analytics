import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, roc_curve,
                              roc_auc_score, accuracy_score)

# ── Fix file paths ─────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_churn.csv')

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📉",
    layout="wide"
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📉 Churn Analytics")
    st.markdown("---")
    st.subheader("⚙️ Model Settings")
    test_size       = st.slider("Test size %", 10, 40, 20) / 100
    churn_threshold = st.slider("Churn threshold %", 30, 70, 50) / 100
    st.markdown("---")
    st.caption(f"Data: {DATA_PATH}")

# ── Load & train ───────────────────────────────────────────────────────────────
@st.cache_data
def load_and_train(test_size):
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        st.error(f"❌ File not found: {DATA_PATH}")
        st.stop()

    # Fix missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    X = df.drop(columns=['Churn']).astype(float)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler        = StandardScaler()
    X_train_s     = np.nan_to_num(scaler.fit_transform(X_train))
    X_test_s      = np.nan_to_num(scaler.transform(X_test))

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)

    y_pred       = model.predict(X_test_s)
    y_pred_proba = model.predict_proba(X_test_s)[:,1]

    return df, X, y, X_test, y_test, y_pred, y_pred_proba, model

# ── Main ───────────────────────────────────────────────────────────────────────
st.title("📉 Customer Churn Analytics Dashboard")
st.markdown("*Predict which customers are likely to leave*")
st.markdown("---")

with st.spinner("⚙️ Training model... please wait..."):
    df, X, y, X_test, y_test, y_pred, y_pred_proba, model = load_and_train(test_size)

y_pred_thresh = (y_pred_proba >= churn_threshold).astype(int)

# ── KPI Cards ──────────────────────────────────────────────────────────────────
accuracy  = accuracy_score(y_test, y_pred_thresh)
roc_score = roc_auc_score(y_test, y_pred_proba)
churn_rate = df['Churn'].mean() * 100
high_risk  = (y_pred_proba >= 0.7).sum()

k1, k2, k3, k4 = st.columns(4)
k1.metric("👥 Total Customers",    f"{len(df):,}")
k2.metric("📉 Churn Rate",         f"{churn_rate:.1f}%")
k3.metric("🎯 Model Accuracy",     f"{accuracy*100:.1f}%")
k4.metric("⚠️ High Risk Customers", f"{high_risk:,}")

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Churn Overview",
    "🤖 Model Performance",
    "⚠️ Risk Analysis",
    "🔍 Feature Importance",
    "📋 Customer Table"
])

# ── Tab 1 ──────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Churn Distribution")
    col1, col2 = st.columns(2)
    with col1:
        churn_counts = df['Churn'].value_counts().reset_index()
        churn_counts.columns = ['Churn','Count']
        churn_counts['Churn'] = churn_counts['Churn'].map({1:'Churned',0:'Stayed'})
        fig = px.pie(churn_counts, values='Count', names='Churn',
                     color_discrete_map={'Churned':'#e74c3c','Stayed':'#2ecc71'},
                     title='Overall Churn Distribution')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        df_temp = df.copy()
        df_temp['Tenure_Group'] = pd.cut(df_temp['tenure'],
                                          bins=[0,12,24,48,72],
                                          labels=['0-12m','13-24m','25-48m','49-72m'])
        tenure_churn = df_temp.groupby(
            'Tenure_Group', observed=True)['Churn'].mean()*100
        fig2 = px.bar(x=tenure_churn.index.astype(str),
                      y=tenure_churn.values,
                      color=tenure_churn.values,
                      color_continuous_scale='RdYlGn_r',
                      title='Churn Rate by Tenure Group',
                      labels={'x':'Tenure','y':'Churn Rate %'})
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.histogram(df,
                         x='MonthlyCharges',
                         color=df['Churn'].map({1:'Churned',0:'Stayed'}),
                         color_discrete_map={'Churned':'#e74c3c','Stayed':'#2ecc71'},
                         barmode='overlay', opacity=0.7,
                         title='Monthly Charges by Churn Status')
    st.plotly_chart(fig3, use_container_width=True)

# ── Tab 2 ──────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Model Evaluation")
    col1, col2 = st.columns(2)
    with col1:
        cm = confusion_matrix(y_test, y_pred_thresh)
        fig_cm = px.imshow(cm,
                            labels=dict(x="Predicted", y="Actual"),
                            x=['Stayed','Churned'],
                            y=['Stayed','Churned'],
                            color_continuous_scale='Blues',
                            title='Confusion Matrix',
                            text_auto=True)
        st.plotly_chart(fig_cm, use_container_width=True)
    with col2:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines',
            name=f'AUC = {roc_score:.3f}',
            line=dict(color='#2E75B6', width=2),
            fill='tozeroy', fillcolor='rgba(46,117,182,0.1)'))
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode='lines',
            name='Random', line=dict(dash='dash', color='gray')))
        fig_roc.update_layout(title='ROC Curve',
                               xaxis_title='False Positive Rate',
                               yaxis_title='True Positive Rate')
        st.plotly_chart(fig_roc, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{accuracy*100:.1f}%")
    m2.metric("ROC-AUC",  f"{roc_score:.3f}")
    m3.metric("Test Size", f"{len(y_test):,} customers")

# ── Tab 3 ──────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Customer Risk Segmentation")

    def risk_level(p):
        if p >= 0.7:   return '🔴 High Risk'
        elif p >= 0.4: return '🟡 Medium Risk'
        else:          return '🟢 Low Risk'

    risk_df = pd.DataFrame({
        'Churn_Probability': y_pred_proba,
        'Actual_Churn'     : y_test.values
    })
    risk_df['Risk_Level'] = risk_df['Churn_Probability'].apply(risk_level)

    col1, col2 = st.columns(2)
    with col1:
        risk_counts = risk_df['Risk_Level'].value_counts().reset_index()
        risk_counts.columns = ['Risk','Count']
        fig_risk = px.bar(risk_counts, x='Risk', y='Count',
                           color='Risk',
                           color_discrete_map={
                               '🔴 High Risk'  :'#e74c3c',
                               '🟡 Medium Risk':'#f39c12',
                               '🟢 Low Risk'   :'#2ecc71'},
                           title='Customer Risk Distribution')
        st.plotly_chart(fig_risk, use_container_width=True)
    with col2:
        fig_prob = px.histogram(risk_df, x='Churn_Probability',
                                 nbins=30,
                                 color_discrete_sequence=['#2E75B6'],
                                 title='Churn Probability Distribution')
        fig_prob.add_vline(x=0.5, line_dash='dash',
                            line_color='red',
                            annotation_text='Threshold')
        st.plotly_chart(fig_prob, use_container_width=True)

    risk_summary = risk_df.groupby('Risk_Level').agg(
        Customers     =('Churn_Probability','count'),
        Avg_Probability=('Churn_Probability','mean'),
        Actually_Churned=('Actual_Churn','sum')
    ).round(3).reset_index()
    st.dataframe(risk_summary, use_container_width=True)

# ── Tab 4 ──────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("What Causes Customers to Churn?")
    feat_df = pd.DataFrame({
        'Feature'    : X.columns,
        'Coefficient': model.coef_[0]
    }).sort_values('Coefficient', ascending=True)

    fig_feat = px.bar(feat_df, x='Coefficient', y='Feature',
                       orientation='h',
                       color='Coefficient',
                       color_continuous_scale='RdYlGn_r',
                       title='Feature Importance')
    st.plotly_chart(fig_feat, use_container_width=True)

    st.subheader("Top 5 Churn Drivers")
    top5 = feat_df.tail(5)[['Feature','Coefficient']].iloc[::-1]
    top5['Impact'] = top5['Coefficient'].apply(
        lambda x: '🔴 Increases churn' if x > 0 else '🟢 Decreases churn')
    st.dataframe(top5, use_container_width=True)

# ── Tab 5 ──────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Customer Predictions Table")

    results = X_test.copy()
    results['Actual_Churn']      = y_test.values
    results['Churn_Probability'] = (y_pred_proba * 100).round(1)
    results['Predicted_Churn']   = y_pred_thresh
    results['Risk_Level']        = y_pred_proba
    results['Risk_Level']        = results['Risk_Level'].apply(risk_level)

    risk_filter = st.selectbox("Filter by risk:",
                                ['All','🔴 High Risk',
                                 '🟡 Medium Risk','🟢 Low Risk'])
    if risk_filter != 'All':
        results = results[results['Risk_Level'] == risk_filter]

    st.markdown(f"Showing **{len(results):,}** customers")

    display_cols = ['tenure','MonthlyCharges','TotalCharges',
                    'Actual_Churn','Churn_Probability','Risk_Level']
    st.dataframe(
        results[display_cols].sort_values(
            'Churn_Probability', ascending=False),
        use_container_width=True, height=400)

    csv = results[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download predictions CSV",
        data=csv,
        file_name="churn_predictions.csv",
        mime="text/csv"
    )