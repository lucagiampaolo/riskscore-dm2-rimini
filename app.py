"""
RiskScore DM2 Rimini — App Streamlit V8
Registro Diabetologico AUSL Romagna | 2026
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle, os, json, warnings
warnings.filterwarnings('ignore')

# ── CONFIGURAZIONE PAGINA ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="RiskScore DM2 — Rimini 2026",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS CUSTOM ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Header principale */
    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border-left: 5px solid #00d4aa;
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 600;
        margin: 0;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #94a3b8;
        margin: 0.3rem 0 0 0;
        font-size: 0.85rem;
    }

    /* Score card */
    .score-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.4rem 0;
        border-left: 4px solid #cbd5e1;
        transition: all 0.2s;
    }
    .score-card:hover { border-left-color: #00d4aa; background: #f0fdf9; }
    .score-card.basso    { border-left-color: #22c55e; background: #f0fdf4; }
    .score-card.medio    { border-left-color: #f59e0b; background: #fffbeb; }
    .score-card.alto     { border-left-color: #ef4444; background: #fef2f2; }

    .score-label { font-size: 0.80rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .score-value { font-size: 1.6rem; font-weight: 600; font-family: 'IBM Plex Mono', monospace; color: #1e293b; }
    .score-risk  { font-size: 0.75rem; color: #64748b; margin-top: 0.2rem; }

    /* Barra progresso custom */
    .progress-bar-bg {
        background: #e2e8f0;
        border-radius: 20px;
        height: 8px;
        margin-top: 0.4rem;
        overflow: hidden;
    }
    .progress-bar-fill {
        height: 100%;
        border-radius: 20px;
        transition: width 0.5s ease;
    }

    /* Tabella risultati */
    .result-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    .result-table th {
        background: #1e293b;
        color: #f1f5f9;
        padding: 0.6rem 0.8rem;
        text-align: left;
        font-weight: 600;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .result-table td { padding: 0.5rem 0.8rem; border-bottom: 1px solid #f1f5f9; }
    .result-table tr:hover td { background: #f8fafc; }

    /* Alert clinico */
    .alert-alto   { background:#fef2f2; border:1px solid #fca5a5; border-radius:8px; padding:0.8rem 1rem; color:#991b1b; }
    .alert-medio  { background:#fffbeb; border:1px solid #fcd34d; border-radius:8px; padding:0.8rem 1rem; color:#92400e; }
    .alert-basso  { background:#f0fdf4; border:1px solid #86efac; border-radius:8px; padding:0.8rem 1rem; color:#166534; }

    /* Metric box */
    div[data-testid="metric-container"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.8rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #1e293b; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label { color: #94a3b8 !important; font-size: 0.82rem; }

    /* Footer */
    .footer { text-align: center; color: #94a3b8; font-size: 0.75rem; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ── CARICO MODELLI ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_params():
    """Carica parametri preprocessing."""
    path = 'params_v2.pkl'
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_resource
def load_dnn():
    """Carica modello DNN."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model

        def masked_bce(y_true, y_pred):
            mask = tf.cast(y_true >= 0, tf.float32)
            y_clean = tf.clip_by_value(y_true, 0, 1)
            loss = tf.keras.losses.binary_crossentropy(y_clean, y_pred)
            return tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-8)

        def masked_mse(y_true, y_pred):
            mask = tf.cast(y_true >= 0, tf.float32)
            loss = tf.square(y_true - y_pred)
            return tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-8)

        model = load_model('mtl_v2_best.h5',
                           custom_objects={'masked_bce': masked_bce,
                                           'masked_mse': masked_mse})
        return model
    except Exception as e:
        return None

@st.cache_resource
def load_xgb_models():
    """Carica modelli XGBoost per ogni outcome."""
    try:
        import xgboost as xgb
        models = {}
        xgb_dir = 'XGB_models/'
        if os.path.isdir(xgb_dir):
            for fname in os.listdir(xgb_dir):
                if fname.endswith('.json'):
                    outcome = fname.replace('xgb_', '').replace('.json', '')
                    m = xgb.XGBClassifier()
                    try:
                        m.load_model(xgb_dir + fname)
                        models[outcome] = m
                    except Exception:
                        try:
                            m2 = xgb.XGBRegressor()
                            m2.load_model(xgb_dir + fname)
                            models[outcome] = m2
                        except Exception:
                            pass
        return models
    except Exception:
        return {}

# ── FUNZIONI CALCOLO ───────────────────────────────────────────────────────────
def score_punti(hba1c, terapia_ipo, dcsi, durata_dm):
    """Formula Score a Punti (4V, 0-16)."""
    pts  = 0 if hba1c <= 56 else (2 if hba1c <= 75 else 4)
    pts += 4 if terapia_ipo else 0
    pts += 0 if dcsi == 0 else (2 if dcsi == 1 else 4)
    pts += 0 if durata_dm <= 2 else (2 if durata_dm <= 10 else 4)
    return pts

def get_risk_class(v):
    if v < 0.25: return "basso",  "🟢 BASSO",      "#22c55e"
    if v < 0.50: return "medio",  "🟡 MEDIO-BASSO", "#f59e0b"
    if v < 0.75: return "alto",   "🟠 MEDIO-ALTO",  "#f97316"
    return              "alto",   "🔴 ALTO",         "#ef4444"

def preprocess_patient(paziente, params):
    """Preprocessing paziente singolo."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    features = params.get('FEATURES_V2', [
        'EtaBasale','HbA1cBasale','eGFR_basale','Albuminuria_basale',
        'DrugScore_basale','TerapiaRischioIpo','Insulina_multiniettiva_basale',
        'DurataDM','TOD_ER_max_basale','ASCVD_basale',
    ])

    vals = [float(paziente.get(f, np.nan)) for f in features]
    X    = np.array([vals], dtype=np.float32)

    imputer = params.get('imputer_v2')
    scaler  = params.get('scaler_v2')

    if imputer is not None:
        X = imputer.transform(X)
    if scaler is not None:
        X = scaler.transform(X)

    return X.astype(np.float32), features

def calcola_dnn_scores(X, model):
    """Calcola score DNN per un paziente."""
    if model is None:
        return {}
    try:
        preds = model.predict(X, verbose=0)
        preds = preds if isinstance(preds, list) else [preds]
        output_names = [out.name.split('/')[0] for out in model.outputs]
        scores = {}
        for i, name in enumerate(output_names):
            if name.startswith('Score_'):
                scores[name] = float(preds[i][0][0])
        return scores
    except Exception:
        return {}

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏥 RiskScore DM2 — Rimini 2026</h1>
    <p>Registro Diabetologico AUSL Romagna &nbsp;|&nbsp; n=12.526 pazienti &nbsp;|&nbsp; Validato 2026 &nbsp;|&nbsp; Uso clinico interno</p>
</div>
""", unsafe_allow_html=True)

# ── CARICO RISORSE ─────────────────────────────────────────────────────────────
params     = load_params()
model_dnn  = load_dnn()
xgb_models = load_xgb_models()

# Status risorse
col_s1, col_s2, col_s3 = st.columns(3)
col_s1.metric("Parametri",    "✅ Caricati"   if params    else "⚠️ Mancanti")
col_s2.metric("Modello DNN",  "✅ Caricato"   if model_dnn else "⚠️ Non disponibile")
col_s3.metric("Modelli XGB",  f"✅ {len(xgb_models)} outcome" if xgb_models else "⚠️ Non disponibili")

st.divider()

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🧑‍⚕️ Paziente Singolo",
    "📊 Lista Pazienti (Excel)",
    "⚖️ Confronto Due Pazienti"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PAZIENTE SINGOLO
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── SIDEBAR INPUT ──────────────────────────────────────────────────────────
    st.sidebar.markdown("## 📋 Dati Paziente")

    with st.sidebar.expander("🧬 Anagrafica", expanded=True):
        eta      = st.number_input("Età (anni)",        18, 100, 65, key="eta")
        durata   = st.number_input("Durata DM (anni)",   0,  50,  8, key="dur")

    with st.sidebar.expander("🔬 Metabolici / Renali", expanded=True):
        hba1c    = st.slider("HbA1c (mmol/mol)", 30, 130, 68, key="hba")
        egfr     = st.slider("eGFR (ml/min)",    10, 150, 75, key="egfr")
        alb      = st.selectbox("Albuminuria",
                                ["Normoalbuminuria (0)",
                                 "Microalbuminuria (1)",
                                 "Macroalbuminuria (2)"], key="alb")
        alb_val  = int(alb.split('(')[1][0])

    with st.sidebar.expander("💊 Terapia", expanded=True):
        drug_sc     = st.slider("DrugScore basale",   0, 60, 20, key="ds")
        terapia_ipo = st.checkbox("Terapia rischio ipoglicemia (ins./SU)", key="tip")
        insulina_mi = st.checkbox("Insulina multi-iniettiva", key="imi")

    with st.sidebar.expander("🫀 Danno d'organo basale", expanded=True):
        tod_max  = st.selectbox("TOD (organi colpiti)",
                                ["0 — Nessuno", "1 — Un organo", "2 — Due o più"], key="tod")
        tod_val  = int(tod_max[0])
        ascvd    = st.checkbox("Evento CV accertato (IMA, ictus, PAD...)", key="asc")

    paziente = {
        'EtaBasale':                    eta,
        'HbA1cBasale':                  hba1c,
        'eGFR_basale':                  egfr,
        'Albuminuria_basale':           alb_val,
        'DrugScore_basale':             drug_sc,
        'TerapiaRischioIpo':            int(terapia_ipo),
        'Insulina_multiniettiva_basale': int(insulina_mi),
        'DurataDM':                     durata,
        'TOD_ER_max_basale':            tod_val,
        'ASCVD_basale':                 int(ascvd),
    }

    # ── CALCOLO SCORE ──────────────────────────────────────────────────────────
    sp = score_punti(hba1c, terapia_ipo, int(alb_val > 0), durata)

    if params is not None:
        X_pz, feats = preprocess_patient(paziente, params)
        dnn_scores  = calcola_dnn_scores(X_pz, model_dnn)
    else:
        X_pz       = None
        dnn_scores = {}

    # ── LAYOUT RISULTATI ───────────────────────────────────────────────────────
    col_main, col_side = st.columns([3, 2])

    with col_main:
        st.subheader("Score a Punti (4V)")

        # Gauge
        sp_pct = sp / 16
        css_cl, risk_lab, risk_col = get_risk_class(sp_pct)
        cat_label = "🟢 BASSO — MMG diretta" if sp <= 5 else \
                    "🟡 MEDIO — MMG + follow-up 6 mesi" if sp <= 10 else \
                    "🔴 ALTO — Diabetologo specialista"

        st.markdown(f"""
        <div class="score-card {css_cl}">
            <div class="score-label">Score a Punti (HbA1c + TerapiaIpo + Albuminuria + DurataDM)</div>
            <div class="score-value">{sp} <span style="font-size:1rem;color:#64748b">/ 16</span></div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill" style="width:{sp_pct*100:.0f}%;
                     background:{'#22c55e' if sp<=5 else '#f59e0b' if sp<=10 else '#ef4444'};">
                </div>
            </div>
            <div class="score-risk" style="font-size:0.9rem;font-weight:600;margin-top:0.5rem;">
                {cat_label}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Dettaglio punti
        with st.expander("Dettaglio punteggio"):
            cols_d = st.columns(4)
            breakdown = {
                "HbA1c":        0 if hba1c<=56 else (2 if hba1c<=75 else 4),
                "Albuminuria":  alb_val * 2,
                "Durata DM":    0 if durata<=2 else (2 if durata<=10 else 4),
                "Terapia Ipo":  4 if terapia_ipo else 0,
            }
            for col_d, (k, v) in zip(cols_d, breakdown.items()):
                col_d.metric(k, f"{v} pt")

        # Score DNN
        if dnn_scores:
            st.subheader("Score per Dominio (DNN)")
            DOMINI_LABELS = {
                'Score_A_Glicemico':    'Controllo Glicemico',
                'Score_B_Terapia':      'Terapia',
                'Score_C_Qualita':      'Qualita Cura',
                'Score_D_Percorso':     'Percorso MMG',
                'Score_E_Complicanze':  'Complicanze',
                'Score_F_Cardiovascolare': 'Cardiovascolare',
                'Score_G_Mortalita':    'Mortalita',
            }
            for sc_name, label in DOMINI_LABELS.items():
                if sc_name in dnn_scores:
                    v = dnn_scores[sc_name]
                    css_c, rl, rc = get_risk_class(v)
                    bar_w = int(v * 100)
                    st.markdown(f"""
                    <div class="score-card {css_c}">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <div>
                                <div class="score-label">{label}</div>
                                <div class="score-risk">{rl}</div>
                            </div>
                            <div class="score-value" style="font-size:1.3rem;">{v:.3f}</div>
                        </div>
                        <div class="progress-bar-bg">
                            <div class="progress-bar-fill" style="width:{bar_w}%;background:{rc};"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # XGBoost outcomes chiave
        if xgb_models and X_pz is not None:
            st.subheader("Predizioni XGBoost — Outcome Chiave")
            xgb_key = {
                'Decesso_finefu': 'Mortalità (follow-up)',
                'Decesso_5a':     'Decesso a 5 anni',
                'DCSI_3a':        'DCSI a 3 anni',
                'MACE5':          'MACE composito 5a',
                'TIR_adj_3a':     'TIR a 3 anni',
                'Pct_tempo_MMG':  'Permanenza MMG',
            }
            xgb_rows = []
            for outcome, label in xgb_key.items():
                if outcome in xgb_models:
                    try:
                        m = xgb_models[outcome]
                        if hasattr(m, 'predict_proba'):
                            pred = float(m.predict_proba(X_pz)[0][1])
                            tipo = 'P(%)'
                            val_str = f"{pred*100:.1f}%"
                        else:
                            pred = float(m.predict(X_pz)[0])
                            tipo = 'Valore'
                            val_str = f"{pred:.3f}"
                        xgb_rows.append({'Outcome': label, 'Tipo': tipo, 'Valore': val_str, '_pred': pred})
                    except Exception:
                        pass

            if xgb_rows:
                xgb_cols = st.columns(3)
                for i, row in enumerate(xgb_rows):
                    with xgb_cols[i % 3]:
                        st.metric(row['Outcome'], row['Valore'])

    with col_side:
        st.subheader("Raccomandazione Clinica")

        if sp <= 5:
            st.markdown("""<div class="alert-basso">
                <b>🟢 Gestione MMG</b><br>
                Paziente eleggibile per gestione integrata col MMG.<br>
                Follow-up annuale sufficiente.
            </div>""", unsafe_allow_html=True)
        elif sp <= 10:
            st.markdown("""<div class="alert-medio">
                <b>🟡 Monitoraggio Attivo</b><br>
                Valutazione diabetologica consigliata entro 6 mesi.<br>
                Follow-up semestrale con MMG.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="alert-alto">
                <b>🔴 Diabetologo Urgente</b><br>
                Invio a specialista. Screening complicanze completo.<br>
                Follow-up trimestrale minimo.
            </div>""", unsafe_allow_html=True)

        st.divider()

        # Riepilogo valori
        st.markdown("**Valori inseriti:**")
        vals_display = {
            "Età": f"{eta} anni",
            "HbA1c": f"{hba1c} mmol/mol",
            "eGFR": f"{egfr} ml/min",
            "Albuminuria": ["Normo","Micro","Macro"][alb_val],
            "DrugScore": str(drug_sc),
            "Durata DM": f"{durata} anni",
            "TOD": f"{tod_val} organi",
            "ASCVD": "Sì" if ascvd else "No",
            "Ipo rischio": "Sì" if terapia_ipo else "No",
        }
        for k, v in vals_display.items():
            st.markdown(f"<small style='color:#64748b'>{k}:</small> **{v}**",
                        unsafe_allow_html=True)

        # Download scheda
        st.divider()
        scheda = pd.DataFrame([{**paziente, 'Score_Punti': sp}])
        st.download_button(
            "⬇️ Scarica scheda paziente",
            scheda.to_csv(index=False).encode('utf-8'),
            "scheda_paziente.csv",
            "text/csv",
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LISTA PAZIENTI
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Calcolo Score su Lista Pazienti")

    st.info("""
    **Formato richiesto:**
    Il file Excel deve contenere le colonne:
    `EtaBasale`, `HbA1cBasale`, `eGFR_basale`, `Albuminuria_basale`,
    `DrugScore_basale`, `TerapiaRischioIpo`, `Insulina_multiniettiva_basale`,
    `DurataDM`, `TOD_ER_max_basale`, `ASCVD_basale`

    Colonne aggiuntive (CF, Baseline, ecc.) vengono mantenute nell'output.
    """)

    uploaded = st.file_uploader("Carica file Excel (.xlsx)", type=['xlsx','xls'])

    if uploaded is not None:
        try:
            df_up = pd.read_excel(uploaded)
            st.success(f"✓ Caricati {len(df_up):,} pazienti × {df_up.shape[1]} colonne")

            FEATURES = [
                'EtaBasale','HbA1cBasale','eGFR_basale','Albuminuria_basale',
                'DrugScore_basale','TerapiaRischioIpo','Insulina_multiniettiva_basale',
                'DurataDM','TOD_ER_max_basale','ASCVD_basale',
            ]
            missing_cols = [c for c in FEATURES if c not in df_up.columns]
            if missing_cols:
                st.warning(f"⚠️ Colonne mancanti: {missing_cols}")

            # Score a Punti per tutti
            def sp_row(row):
                hba = float(row.get('HbA1cBasale', 65))
                tip = bool(row.get('TerapiaRischioIpo', 0))
                alb = float(row.get('Albuminuria_basale', 0))
                dur = float(row.get('DurataDM', 5))
                return score_punti(hba, tip, int(alb > 0), dur)

            df_up['Score_Punti'] = df_up.apply(sp_row, axis=1)
            df_up['Categoria']   = df_up['Score_Punti'].apply(
                lambda x: 'BASSO' if x<=5 else ('MEDIO' if x<=10 else 'ALTO')
            )

            # Score DNN se disponibile
            if params is not None and model_dnn is not None:
                feats_avail = [f for f in FEATURES if f in df_up.columns]
                if feats_avail:
                    with st.spinner("Calcolo score DNN..."):
                        from sklearn.impute import SimpleImputer
                        from sklearn.preprocessing import StandardScaler
                        X_batch = df_up[feats_avail].apply(pd.to_numeric, errors='coerce')
                        imp = params.get('imputer_v2')
                        scl = params.get('scaler_v2')
                        if imp and scl:
                            X_proc = scl.transform(imp.transform(X_batch)).astype(np.float32)
                            preds  = model_dnn.predict(X_proc, verbose=0)
                            preds  = preds if isinstance(preds, list) else [preds]
                            out_names = [o.name.split('/')[0] for o in model_dnn.outputs]
                            for i, name in enumerate(out_names):
                                if name.startswith('Score_'):
                                    df_up[name] = preds[i].flatten()
                    st.success("✓ Score DNN calcolati")

            # Anteprima
            st.subheader("Anteprima risultati")
            score_cols = ['Score_Punti','Categoria'] + \
                         [c for c in df_up.columns if c.startswith('Score_') and c != 'Score_Punti']
            id_cols    = [c for c in ['CF','Baseline','EtaBasale','HbA1cBasale'] if c in df_up.columns]
            st.dataframe(df_up[id_cols + score_cols].head(20), use_container_width=True)

            # Distribuzione
            col_d1, col_d2, col_d3 = st.columns(3)
            n_basso = (df_up['Categoria']=='BASSO').sum()
            n_medio = (df_up['Categoria']=='MEDIO').sum()
            n_alto  = (df_up['Categoria']=='ALTO').sum()
            col_d1.metric("🟢 BASSO",  f"{n_basso:,}", f"{n_basso/len(df_up)*100:.1f}%")
            col_d2.metric("🟡 MEDIO",  f"{n_medio:,}", f"{n_medio/len(df_up)*100:.1f}%")
            col_d3.metric("🔴 ALTO",   f"{n_alto:,}",  f"{n_alto/len(df_up)*100:.1f}%")

            # Download
            st.download_button(
                "⬇️ Scarica risultati completi",
                df_up.to_csv(index=False).encode('utf-8'),
                "risultati_score.csv",
                "text/csv",
            )

        except Exception as e:
            st.error(f"Errore caricamento: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CONFRONTO
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Confronto Due Pazienti")
    st.caption("Inserisci i valori di due pazienti per confrontare il profilo di rischio")

    col_p1, col_sep, col_p2 = st.columns([5, 1, 5])

    def paziente_form(prefix, col):
        with col:
            st.markdown(f"**Paziente {prefix}**")
            h = st.number_input("HbA1c",    30, 130, 68, key=f"h_{prefix}")
            e = st.number_input("Età",       18, 100, 65, key=f"e_{prefix}")
            g = st.number_input("eGFR",      10, 150, 75, key=f"g_{prefix}")
            a = st.selectbox("Albuminuria", [0,1,2], key=f"a_{prefix}")
            d = st.number_input("Durata DM",  0,  50,  8, key=f"d_{prefix}")
            t = st.number_input("TOD",         0,   2,  0, key=f"t_{prefix}")
            c = st.checkbox("CV accertato", key=f"c_{prefix}")
            tip = st.checkbox("Ipo rischio", key=f"tip_{prefix}")
            return {'HbA1cBasale':h,'EtaBasale':e,'eGFR_basale':g,
                    'Albuminuria_basale':a,'DurataDM':d,
                    'TOD_ER_max_basale':t,'ASCVD_basale':int(c),
                    'TerapiaRischioIpo':int(tip),
                    'Insulina_multiniettiva_basale':0,
                    'DrugScore_basale':20}

    p1 = paziente_form("A", col_p1)
    with col_sep:
        st.markdown("<br><br><br><br><br><br><br>VS", unsafe_allow_html=True)
    p2 = paziente_form("B", col_p2)

    sp1 = score_punti(p1['HbA1cBasale'], bool(p1['TerapiaRischioIpo']),
                      int(p1['Albuminuria_basale']>0), p1['DurataDM'])
    sp2 = score_punti(p2['HbA1cBasale'], bool(p2['TerapiaRischioIpo']),
                      int(p2['Albuminuria_basale']>0), p2['DurataDM'])

    st.divider()
    st.subheader("Confronto Score a Punti")

    col_r1, col_r2 = st.columns(2)
    for col_r, sp, label in [(col_r1, sp1, "Paziente A"), (col_r2, sp2, "Paziente B")]:
        with col_r:
            css_c, rl, rc = get_risk_class(sp/16)
            st.markdown(f"""
            <div class="score-card {css_c}" style="text-align:center;">
                <div class="score-label">{label}</div>
                <div class="score-value" style="font-size:2.5rem;">{sp}<span style="font-size:1rem;color:#64748b">/16</span></div>
                <div class="score-risk" style="font-size:1rem;font-weight:600;">{rl}</div>
            </div>
            """, unsafe_allow_html=True)

    if sp1 != sp2:
        diff = abs(sp1 - sp2)
        more = "Paziente A" if sp1 > sp2 else "Paziente B"
        st.info(f"**{more}** ha uno score {diff} punti più alto.")
    else:
        st.info("I due pazienti hanno lo stesso Score a Punti.")

    # DNN confronto
    if params is not None and model_dnn is not None:
        DOMINI_LABELS = {
            'Score_A_Glicemico': 'Glicemico',
            'Score_D_Percorso':  'Percorso MMG',
            'Score_E_Complicanze': 'Complicanze',
            'Score_F_Cardiovascolare': 'CV',
            'Score_G_Mortalita': 'Mortalita',
        }
        X1, _ = preprocess_patient(p1, params)
        X2, _ = preprocess_patient(p2, params)
        s1 = calcola_dnn_scores(X1, model_dnn)
        s2 = calcola_dnn_scores(X2, model_dnn)

        if s1 and s2:
            st.subheader("Confronto Score DNN per Dominio")
            compare_data = []
            for sc_name, label in DOMINI_LABELS.items():
                v1 = s1.get(sc_name, np.nan)
                v2 = s2.get(sc_name, np.nan)
                if not np.isnan(v1) and not np.isnan(v2):
                    compare_data.append({'Dominio': label,
                                         'Paziente A': v1,
                                         'Paziente B': v2,
                                         'Delta (A-B)': round(v1-v2, 3)})
            if compare_data:
                df_cmp = pd.DataFrame(compare_data)
                st.dataframe(df_cmp.style.background_gradient(
                    subset=['Paziente A','Paziente B'], cmap='RdYlGn_r'),
                    use_container_width=True)

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    RiskScore DM2 Rimini 2026 &nbsp;|&nbsp; AUSL Romagna &nbsp;|&nbsp;
    n=12.526 pazienti DM2 &nbsp;|&nbsp; AUC mortalità 0.821 (XGBoost) &nbsp;|&nbsp;
    <b>Uso esclusivo per supporto clinico interno. Non sostituisce il giudizio del medico.</b>
</div>
""", unsafe_allow_html=True)
