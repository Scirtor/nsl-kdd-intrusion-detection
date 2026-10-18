"""Local CSV demonstration using only a locally generated trusted model."""
import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from nslkdd.data import ROOT, validate_features

st.set_page_config(page_title='NSL-KDD Intrusion Detection', page_icon='🔎', layout='wide')
st.title('Network Intrusion Detection')
st.caption('Academic benchmark · NSL-KDD · Normal versus attack')
st.write('Explore a machine learning classifier on precomputed connection features. This demo does not capture live network traffic.')
run_dir = ROOT / 'runs/baseline'
if not (run_dir / 'model.joblib').exists():
    st.info('Train the baseline first: python -m nslkdd.train')
    st.stop()


@st.cache_resource
def load_model(path, modified):
    return joblib.load(path)


model_path=run_dir/'model.joblib'
model=load_model(model_path,model_path.stat().st_mtime_ns)
report=json.loads((run_dir/'metrics.json').read_text())
st.subheader('Official test results')
a,b,c=st.columns(3)
a.metric('Attack recall',f'{report["official_test"]["recall"]:.1%}')
b.metric('Attack F1',f'{report["official_test"]["f1"]:.3f}')
c.metric('False positive rate',f'{report["official_test"]["false_positive_rate"]:.1%}')
st.caption(f'Model: {report["selected_model"]}. Selected using a separate validation subset of KDDTrain+. Scores describe this historical benchmark, not production detection quality.')
st.subheader('Try a feature CSV')
sample=(ROOT/'examples/connections.csv').read_bytes()
st.download_button('Download example CSV',sample,'connections.csv','text/csv')
uploaded=st.file_uploader('Upload a CSV with the 41 NSL-KDD feature columns',type=['csv'])
try:
    frame=pd.read_csv(uploaded) if uploaded is not None else pd.read_csv(ROOT/'examples/connections.csv')
    if len(frame)>10000:
        st.error('Use at most 10,000 rows per file.');st.stop()
    if frame.empty:
        st.error('The file contains no rows.');st.stop()
    X=validate_features(frame)
    result=X.copy()
    result['verdict']=['Attack' if p else 'Normal' for p in model.predict(X)]
    st.dataframe(result,hide_index=True)
    st.download_button('Download predictions',result.to_csv(index=False).encode(),'predictions.csv','text/csv')
except (ValueError,TypeError) as exc:
    st.error(f'Cannot process this CSV: {exc}')
st.caption('Files are processed by this local application. No model upload or external API is used.')
