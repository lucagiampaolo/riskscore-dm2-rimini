"""
RiskScore DM2 Rimini — App Streamlit V8
Registro Diabetologico AUSL Romagna | 2026
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle, os, warnings
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
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 2rem 2.5rem; border-radius: 12px;
        margin-bottom: 1.5rem; border-left: 5px solid #00d4aa;
    }
    .main-header h1 {
        color: #ffffff; font-size: 1.8rem; font-weight: 600; margin: 0;
        font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.5px;
    }
    .main-header p { color: #94a3b8; margin: 0.3rem 0 0 0; font-size: 0.85rem; }

    .score-card {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 1.2rem; margin: 0.4rem 0; border-left: 4px solid #cbd5e1;
    }
    .score-card.basso  { border-left-color: #22c55e; background: #f0fdf4; }
    .score-card.medio  { border-left-color: #f59e0b; background: #fffbeb; }
    .score-card.alto   { border-left-color: #ef4444; background: #fef2f2; }

    .score-label { font-size: 0.80rem; color: #64748b; font-weight: 600;
                   text-transform: uppercase; letter-spacing: 0.5px; }
    .score-value { font-size: 1.6rem; font-weight: 600;
                   font-family: 'IBM Plex Mono', monospace; color: #1e293b; }

    .progress-bar-bg { background: #e2e8f0; border-radius: 20px; height: 8px;
                       margin-top: 0.4rem; overflow: hidden; }
    .progress-bar-fill { height: 100%; border-radius: 20px; }

    .alert-alto  { background:#fef2f2; border:1px solid #fca5a5; border-radius:8px;
                   padding:0.8rem 1rem; color:#991b1b; }
    .alert-medio { background:#fffbeb; border:1px solid #fcd34d; border-radius:8px;
                   padding:0.8rem 1rem; color:#92400e; }
    .alert-basso { background:#f0fdf4; border:1px solid #86efac; border-radius:8px;
                   padding:0.8rem 1rem; color:#166534; }

    div[data-testid="metric-container"] {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 0.8rem;
    }
    [data-testid="stSidebar"] { background: #1e293b; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    .footer { text-align: center; color: #94a3b8; font-size: 0.75rem;
              margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)


# ── CARICO MODELLI ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_params():
    """Carica parametri — robusto a versioni sklearn diverse."""
    FEATURES_DEFAULT = [
        'EtaBasale','HbA1cBasale','eGFR_basale','Albuminuria_basale',
        'DrugScore_basale','TerapiaRischioIpo','Insulina_multiniettiva_basale',
        'DurataDM','TOD_ER_max_basale','ASCVD_basale',
    ]
    path = 'params_v2.pkl'
    if not os.path.isfile(path):
        return {'FEATURES_V2': FEATURES_DEFAULT}
    try:
        with open(path, 'rb') as f:
            params = pickle.load(f)
        # Testa imputer — se fallisce lo disabilita
        imputer = params.get('imputer_v2')
        if imputer is not None:
            test = np.array([[66,66,79,0,18,0,0,7,0,0]], dtype=np.float32)
            try:
                imputer.transform(test)
            except Exception:
                params['imputer_v2'] = None
        # Testa scaler
        scaler = params.get('scaler_v2')
        if scaler is not None:
            try:
                test2 = np.zeros((1, len(params.get('FEATURES_V2', FEATURES_DEFAULT))),
                                 dtype=np.float32)
                scaler.transform(test2)
            except Exception:
                params['scaler_v2'] = None
        if 'FEATURES_V2' not in params:
            params['FEATURES_V2'] = FEATURES_DEFAULT
        return params
    except Exception:
        return {'FEATURES_V2': FEATURES_DEFAULT}


@st.cache_resource
def load_dnn():
    """Carica modello DNN se disponibile."""
    if not os.path.isfile('mtl_v2_best.h5'):
        return None
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model

        def masked_bce(y_true, y_pred):
            mask    = tf.cast(y_true >= 0, tf.float32)
            y_clean = tf.clip_by_value(y_true, 0, 1)
            loss    = tf.keras.losses.binary_crossentropy(y_clean, y_pred)
            return tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-8)

        def masked_mse(y_true, y_pred):
            mask = tf.cast(y_true >= 0, tf.float32)
            loss = tf.square(y_true - y_pred)
            return tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-8)

        return load_model('mtl_v2_best.h5',
                          custom_objects={'masked_bce': masked_bce,
                                          'masked_mse': masked_mse})
    except Exception:
        return None


@st.cache_resource
def load_xgb_models():
    """Carica modelli XGBoost."""
    models = {}
    xgb_dir = 'XGB_models/'
    if not os.path.isdir(xgb_dir):
        return models
    try:
        import xgboost as xgb
        for fname in sorted(os.listdir(xgb_dir)):
            if not fname.endswith('.json'):
                continue
            outcome = fname.replace('xgb_', '').replace('.json', '')
            path_m  = xgb_dir + fname
            # Prova classificatore poi regressore
            for cls in [xgb.XGBClassifier, xgb.XGBRegressor]:
                try:
                    m = cls()
                    m.load_model(path_m)
                    models[outcome] = m
                    break
                except Exception:
                    pass
    except Exception:
        pass
    return models


# ── FUNZIONI CALCOLO ───────────────────────────────────────────────────────────
def score_punti(hba1c, terapia_ipo, albuminuria, durata_dm):
    """Score a Punti (4V, 0-16)."""
    pts  = 0 if hba1c <= 56 else (2 if hba1c <= 75 else 4)
    pts += 4 if terapia_ipo else 0
    pts += 0 if albuminuria == 0 else (2 if albuminuria == 1 else 4)
    pts += 0 if durata_dm <= 2 else (2 if durata_dm <= 10 else 4)
    return pts


def get_risk_class(v):
    if v < 0.25: return "basso", "🟢 BASSO",       "#22c55e"
    if v < 0.50: return "medio", "🟡 MEDIO-BASSO",  "#f59e0b"
    if v < 0.75: return "alto",  "🟠 MEDIO-ALTO",   "#f97316"
    return              "alto",  "🔴 ALTO",          "#ef4444"


def preprocess_patient(paziente, params):
    """Preprocessing robusto — non dipende da versione sklearn."""
    features = params.get('FEATURES_V2', [
        'EtaBasale','HbA1cBasale','eGFR_basale','Albuminuria_basale',
        'DrugScore_basale','TerapiaRischioIpo','Insulina_multiniettiva_basale',
        'DurataDM','TOD_ER_max_basale','ASCVD_basale',
    ])

    # Mediane popolazione DM2 Rimini (usate se valore mancante)
    MEDIANE = {
        'EtaBasale': 66.0, 'HbA1cBasale': 66.0, 'eGFR_basale': 79.0,
        'Albuminuria_basale': 0.0, 'DrugScore_basale': 18.0,
        'TerapiaRischioIpo': 0.0, 'Insulina_multiniettiva_basale': 0.0,
        'DurataDM': 7.0, 'TOD_ER_max_basale': 0.0, 'ASCVD_basale': 0.0,
    }

    vals = [float(paziente.get(f, np.nan)) for f in features]
    X    = np.array([vals], dtype=np.float32)

    # Imputa NaN con mediane
    for j, feat in enumerate(features):
        if np.isnan(X[0, j]):
            X[0, j] = float(MEDIANE.get(feat, 0.0))

    # Prova scaler salvato — fallback a standardizzazione manuale
    scaler = params.get('scaler_v2')
    if scaler is not None:
        try:
            X = scaler.transform(X).astype(np.float32)
            return X, features
        except Exception:
            pass

    # Standardizzazione manuale con medie/std tipiche popolazione
    MEANS = np.array([66, 66, 79, 0.3, 18, 0.3, 0.15, 7,  0.4, 0.2])
    STDS  = np.array([11, 16, 22, 0.6, 12, 0.4, 0.35, 6,  0.6, 0.4])
    n     = len(features)
    X     = ((X - MEANS[:n]) / (STDS[:n] + 1e-8)).astype(np.float32)
    return X, features


def calcola_dnn_scores(X, model):
    """Score dai bottleneck DNN."""
    if model is None:
        return {}
    try:
        preds      = model.predict(X, verbose=0)
        preds      = preds if isinstance(preds, list) else [preds]
        out_names  = [o.name.split('/')[0] for o in model.outputs]
        return {name: float(preds[i][0][0])
                for i, name in enumerate(out_names)
                if name.startswith('Score_')}
    except Exception:
        return {}


# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏥 RiskScore DM2 — Rimini 2026</h1>
    <p>Registro Diabetologico AUSL Romagna &nbsp;|&nbsp; n=12.526 pazienti
       &nbsp;|&nbsp; Validato 2026 &nbsp;|&nbsp; Uso clinico interno</p>
</div>
""", unsafe_allow_html=True)

# ── CARICO RISORSE ─────────────────────────────────────────────────────────────
params     = load_params()
model_dnn  = load_dnn()
xgb_models = load_xgb_models()

col_s1, col_s2, col_s3 = st.columns(3)
col_s1.metric("Parametri",   "✅ Caricati"            if params                    else "⚠️ Mancanti")
col_s2.metric("Modello DNN", "✅ Caricato"             if model_dnn                 else "⚠️ Non disponibile")
col_s3.metric("Modelli XGB", f"✅ {len(xgb_models)} outcome" if xgb_models         else "⚠️ Non disponibili")

st.divider()

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🧑‍⚕️ Paziente Singolo",
    "📊 Lista Pazienti (Excel)",
    "⚖️ Confronto Due Pazienti",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PAZIENTE SINGOLO
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # Sidebar
    st.sidebar.markdown("## 📋 Dati Paziente")

    with st.sidebar.expander("🧬 Anagrafica", expanded=True):
        eta    = st.number_input("Età (anni)",       18, 100, 65, key="eta")
        durata = st.number_input("Durata DM (anni)",  0,  50,  8, key="dur")

    with st.sidebar.expander("🔬 Metabolici / Renali", expanded=True):
        hba1c = st.slider("HbA1c (mmol/mol)", 30, 130, 68, key="hba")
        egfr  = st.slider("eGFR (ml/min)",    10, 150, 75, key="egfr")
        alb   = st.selectbox("Albuminuria",
                             ["Normoalbuminuria (0)",
                              "Microalbuminuria (1)",
                              "Macroalbuminuria (2)"], key="alb")
        alb_val = int(alb.split('(')[1][0])

    with st.sidebar.expander("💊 Terapia", expanded=True):
        drug_sc     = st.slider("DrugScore basale",  0, 60, 20, key="ds")
        terapia_ipo = st.checkbox("Terapia rischio ipoglicemia (ins./SU)", key="tip")
        insulina_mi = st.checkbox("Insulina multi-iniettiva", key="imi")

    with st.sidebar.expander("🫀 Danno organo basale", expanded=True):
        tod_sel = st.selectbox("TOD (organi colpiti)",
                               ["0 — Nessuno", "1 — Un organo", "2 — Due o più"],
                               key="tod")
        tod_val = int(tod_sel[0])
        ascvd   = st.checkbox("Evento CV accertato (IMA, ictus, PAD...)", key="asc")

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

    # Calcolo score
    sp = score_punti(hba1c, terapia_ipo, alb_val, durata)
    X_pz, feats = preprocess_patient(paziente, params)
    dnn_scores  = calcola_dnn_scores(X_pz, model_dnn)

    # Layout
    col_main, col_side = st.columns([3, 2])

    with col_main:
        st.subheader("Score a Punti (4V)")

        sp_pct   = sp / 16
        css_cl, risk_lab, risk_col = get_risk_class(sp_pct)
        bar_color = '#22c55e' if sp <= 5 else ('#f59e0b' if sp <= 10 else '#ef4444')
        cat_label = ("🟢 BASSO — MMG diretta"          if sp <= 5  else
                     "🟡 MEDIO — MMG + follow-up 6m"   if sp <= 10 else
                     "🔴 ALTO — Diabetologo specialista")

        st.markdown(f"""
        <div class="score-card {css_cl}">
            <div class="score-label">HbA1c · Albuminuria · TerapiaIpo · DurataDM</div>
            <div class="score-value">{sp}
                <span style="font-size:1rem;color:#64748b"> / 16</span>
            </div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill"
                     style="width:{sp_pct*100:.0f}%;background:{bar_color};"></div>
            </div>
            <div style="font-size:0.95rem;font-weight:600;margin-top:0.5rem;">
                {cat_label}
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Dettaglio punteggio"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("HbA1c",      f"{0 if hba1c<=56 else (2 if hba1c<=75 else 4)} pt")
            c2.metric("Albuminuria", f"{alb_val*2} pt")
            c3.metric("Durata DM",  f"{0 if durata<=2 else (2 if durata<=10 else 4)} pt")
            c4.metric("Terapia Ipo", f"{4 if terapia_ipo else 0} pt")

        # Score DNN bottleneck
        if dnn_scores:
            st.subheader("Score per Dominio — DNN")
            DOMINI_LABELS = {
                'Score_A_Glicemico':      'Controllo Glicemico',
                'Score_B_Terapia':        'Complessità Terapeutica',
                'Score_C_Qualita':        'Qualità della Cura',
                'Score_D_Percorso':       'Percorso MMG',
                'Score_E_Complicanze':    'Complicanze Croniche',
                'Score_F_Cardiovascolare':'Rischio Cardiovascolare',
                'Score_G_Mortalita':      'Mortalità',
            }
            for sc_name, label in DOMINI_LABELS.items():
                if sc_name not in dnn_scores:
                    continue
                v       = dnn_scores[sc_name]
                css_c, rl, rc = get_risk_class(v)
                st.markdown(f"""
                <div class="score-card {css_c}">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <div class="score-label">{label}</div>
                            <div style="font-size:0.85rem;color:{rc};font-weight:600;">{rl}</div>
                        </div>
                        <div class="score-value" style="font-size:1.4rem;">{v:.3f}</div>
                    </div>
                    <div class="progress-bar-bg">
                        <div class="progress-bar-fill"
                             style="width:{v*100:.0f}%;background:{rc};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # XGBoost outcome chiave
        if xgb_models:
            st.subheader("Predizioni XGBoost — Outcome Chiave")
            XGB_KEY = {
                'Decesso_finefu': 'Mortalità (follow-up)',
                'Decesso_5a':     'Decesso a 5 anni',
                'Decesso_3a':     'Decesso a 3 anni',
                'DCSI_3a':        'DCSI a 3 anni',
                'MACE5':          'MACE composito 5a',
                'TIR_adj_3a':     'TIR a 3 anni',
                'Pct_tempo_MMG':  'Tempo in MMG',
                'ASCVD_inc_5a':   'ASCVD inc. 5 anni',
                'TOD_0to1_3a':    'Nuova TOD a 3 anni',
            }
            xgb_rows = []
            for outcome, label in XGB_KEY.items():
                if outcome not in xgb_models:
                    continue
                try:
                    m = xgb_models[outcome]
                    if hasattr(m, 'predict_proba'):
                        pred    = float(m.predict_proba(X_pz)[0][1])
                        val_str = f"{pred*100:.1f}%"
                    else:
                        pred    = float(m.predict(X_pz)[0])
                        val_str = f"{pred:.3f}"
                    xgb_rows.append({'label': label, 'val': val_str, 'pred': pred})
                except Exception:
                    pass

            if xgb_rows:
                cols_xgb = st.columns(3)
                for i, row in enumerate(xgb_rows):
                    cols_xgb[i % 3].metric(row['label'], row['val'])

    with col_side:
        st.subheader("Raccomandazione Clinica")

        if sp <= 5:
            st.markdown("""<div class="alert-basso">
                <b>🟢 Gestione MMG</b><br>
                Eleggibile per gestione integrata MMG.<br>
                Follow-up annuale sufficiente.
            </div>""", unsafe_allow_html=True)
        elif sp <= 10:
            st.markdown("""<div class="alert-medio">
                <b>🟡 Monitoraggio Attivo</b><br>
                Valutazione diabetologica entro 6 mesi.<br>
                Follow-up semestrale con MMG.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="alert-alto">
                <b>🔴 Diabetologo Urgente</b><br>
                Invio specialista. Screening complicanze.<br>
                Follow-up trimestrale minimo.
            </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("**Valori inseriti:**")
        display = {
            "Età": f"{eta} anni",            "HbA1c": f"{hba1c} mmol/mol",
            "eGFR": f"{egfr} ml/min",        "Albuminuria": ["Normo","Micro","Macro"][alb_val],
            "DrugScore": str(drug_sc),        "Durata DM": f"{durata} anni",
            "TOD": f"{tod_val} organi",       "ASCVD": "Sì" if ascvd else "No",
            "Ipo rischio": "Sì" if terapia_ipo else "No",
        }
        for k, v in display.items():
            st.markdown(f"<small style='color:#64748b'>{k}:</small> **{v}**",
                        unsafe_allow_html=True)

        st.divider()
        scheda = pd.DataFrame([{**paziente, 'Score_Punti': sp}])
        st.download_button(
            "⬇️ Scarica scheda paziente",
            scheda.to_csv(index=False).encode('utf-8'),
            "scheda_paziente.csv", "text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LISTA PAZIENTI
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Calcolo Score su Lista Pazienti")

    st.info("""
    **Colonne richieste nel file Excel:**
    `EtaBasale` · `HbA1cBasale` · `eGFR_basale` · `Albuminuria_basale`
    `DrugScore_basale` · `TerapiaRischioIpo` · `Insulina_multiniettiva_basale`
    `DurataDM` · `TOD_ER_max_basale` · `ASCVD_basale`

    Colonne extra (CF, Baseline...) vengono mantenute nell'output.
    """)

    uploaded = st.file_uploader("Carica file Excel", type=['xlsx','xls'])

    if uploaded is not None:
        try:
            df_up = pd.read_excel(uploaded)
            st.success(f"✓ {len(df_up):,} pazienti × {df_up.shape[1]} colonne")

            FEATURES = params.get('FEATURES_V2', [
                'EtaBasale','HbA1cBasale','eGFR_basale','Albuminuria_basale',
                'DrugScore_basale','TerapiaRischioIpo','Insulina_multiniettiva_basale',
                'DurataDM','TOD_ER_max_basale','ASCVD_basale',
            ])
            missing = [c for c in FEATURES if c not in df_up.columns]
            if missing:
                st.warning(f"⚠️ Colonne mancanti: {missing}")

            # Score a Punti
            def sp_row(row):
                return score_punti(
                    float(row.get('HbA1cBasale', 65)),
                    bool(row.get('TerapiaRischioIpo', 0)),
                    int(float(row.get('Albuminuria_basale', 0))),
                    float(row.get('DurataDM', 5))
                )
            df_up['Score_Punti'] = df_up.apply(sp_row, axis=1)
            df_up['Categoria']   = df_up['Score_Punti'].apply(
                lambda x: 'BASSO' if x <= 5 else ('MEDIO' if x <= 10 else 'ALTO')
            )

            # DNN in batch
            if model_dnn is not None:
                feats_ok = [f for f in FEATURES if f in df_up.columns]
                if feats_ok:
                    with st.spinner("Calcolo score DNN..."):
                        MEDIANE = {
                            'EtaBasale':66,'HbA1cBasale':66,'eGFR_basale':79,
                            'Albuminuria_basale':0,'DrugScore_basale':18,
                            'TerapiaRischioIpo':0,'Insulina_multiniettiva_basale':0,
                            'DurataDM':7,'TOD_ER_max_basale':0,'ASCVD_basale':0,
                        }
                        X_batch = df_up[feats_ok].apply(pd.to_numeric, errors='coerce')
                        for col in feats_ok:
                            X_batch[col] = X_batch[col].fillna(MEDIANE.get(col, 0))
                        X_np = X_batch.values.astype(np.float32)

                        scaler = params.get('scaler_v2')
                        if scaler:
                            try:
                                X_np = scaler.transform(X_np).astype(np.float32)
                            except Exception:
                                MEANS = np.array([66,66,79,0.3,18,0.3,0.15,7,0.4,0.2])
                                STDS  = np.array([11,16,22,0.6,12,0.4,0.35,6,0.6,0.4])
                                n = X_np.shape[1]
                                X_np = ((X_np - MEANS[:n]) / (STDS[:n]+1e-8)).astype(np.float32)

                        preds     = model_dnn.predict(X_np, batch_size=256, verbose=0)
                        preds     = preds if isinstance(preds, list) else [preds]
                        out_names = [o.name.split('/')[0] for o in model_dnn.outputs]
                        for i, name in enumerate(out_names):
                            if name.startswith('Score_'):
                                df_up[name] = preds[i].flatten()
                    st.success("✓ Score DNN calcolati")

            # Anteprima
            score_cols = ['Score_Punti','Categoria'] + \
                         [c for c in df_up.columns
                          if c.startswith('Score_') and c != 'Score_Punti']
            id_cols = [c for c in ['CF','Baseline','EtaBasale','HbA1cBasale']
                       if c in df_up.columns]
            st.dataframe(df_up[id_cols + score_cols].head(20),
                         use_container_width=True)

            # Distribuzione
            c1, c2, c3 = st.columns(3)
            for col_m, cat, emoji in [(c1,'BASSO','🟢'),(c2,'MEDIO','🟡'),(c3,'ALTO','🔴')]:
                n = (df_up['Categoria'] == cat).sum()
                col_m.metric(f"{emoji} {cat}", f"{n:,}",
                             f"{n/len(df_up)*100:.1f}%")

            st.download_button(
                "⬇️ Scarica risultati",
                df_up.to_csv(index=False).encode('utf-8'),
                "risultati_score.csv", "text/csv",
            )

        except Exception as e:
            st.error(f"Errore: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CONFRONTO
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Confronto Due Pazienti")

    def paziente_form(prefix, col):
        with col:
            st.markdown(f"**Paziente {prefix}**")
            h   = st.number_input("HbA1c (mmol/mol)", 30, 130, 68, key=f"h_{prefix}")
            e   = st.number_input("Età (anni)",        18, 100, 65, key=f"e_{prefix}")
            g   = st.number_input("eGFR (ml/min)",     10, 150, 75, key=f"g_{prefix}")
            a   = st.selectbox("Albuminuria", [0,1,2],
                               format_func=lambda x: ["Normo","Micro","Macro"][x],
                               key=f"a_{prefix}")
            d   = st.number_input("Durata DM (anni)",   0,  50,  8, key=f"d_{prefix}")
            t   = st.number_input("TOD (0-2)",           0,   2,  0, key=f"t_{prefix}")
            cv  = st.checkbox("CV accertato",  key=f"cv_{prefix}")
            tip = st.checkbox("Ipo rischio",   key=f"tip_{prefix}")
            ds  = st.number_input("DrugScore",  0, 60, 20, key=f"ds_{prefix}")
            return {
                'HbA1cBasale': h, 'EtaBasale': e, 'eGFR_basale': g,
                'Albuminuria_basale': a, 'DurataDM': d,
                'TOD_ER_max_basale': t, 'ASCVD_basale': int(cv),
                'TerapiaRischioIpo': int(tip),
                'Insulina_multiniettiva_basale': 0,
                'DrugScore_basale': ds,
            }

    col_p1, col_sep, col_p2 = st.columns([5, 1, 5])
    p1 = paziente_form("A", col_p1)
    with col_sep:
        st.markdown("<br>"*8 + "**VS**", unsafe_allow_html=True)
    p2 = paziente_form("B", col_p2)

    sp1 = score_punti(p1['HbA1cBasale'], bool(p1['TerapiaRischioIpo']),
                      int(p1['Albuminuria_basale']), p1['DurataDM'])
    sp2 = score_punti(p2['HbA1cBasale'], bool(p2['TerapiaRischioIpo']),
                      int(p2['Albuminuria_basale']), p2['DurataDM'])

    st.divider()
    st.subheader("Score a Punti")
    col_r1, col_r2 = st.columns(2)

    for col_r, sp, label in [(col_r1, sp1, "Paziente A"), (col_r2, sp2, "Paziente B")]:
        with col_r:
            css_c, rl, rc = get_risk_class(sp/16)
            st.markdown(f"""
            <div class="score-card {css_c}" style="text-align:center;">
                <div class="score-label">{label}</div>
                <div class="score-value" style="font-size:2.5rem;">
                    {sp}<span style="font-size:1rem;color:#64748b">/16</span>
                </div>
                <div style="font-size:1rem;font-weight:600;color:{rc};">{rl}</div>
            </div>
            """, unsafe_allow_html=True)

    if sp1 != sp2:
        more = "Paziente A" if sp1 > sp2 else "Paziente B"
        st.info(f"**{more}** ha uno score {abs(sp1-sp2)} punti più alto.")
    else:
        st.success("I due pazienti hanno lo stesso Score a Punti.")

    # Confronto DNN
    if model_dnn is not None:
        X1, _ = preprocess_patient(p1, params)
        X2, _ = preprocess_patient(p2, params)
        s1    = calcola_dnn_scores(X1, model_dnn)
        s2    = calcola_dnn_scores(X2, model_dnn)

        if s1 and s2:
            st.subheader("Confronto Score DNN per Dominio")
            DOMINI_SHORT = {
                'Score_A_Glicemico':      'Glicemico',
                'Score_D_Percorso':       'Percorso MMG',
                'Score_E_Complicanze':    'Complicanze',
                'Score_F_Cardiovascolare':'CV',
                'Score_G_Mortalita':      'Mortalità',
            }
            rows = []
            for sc, label in DOMINI_SHORT.items():
                v1 = s1.get(sc, np.nan)
                v2 = s2.get(sc, np.nan)
                if not (np.isnan(v1) or np.isnan(v2)):
                    rows.append({'Dominio': label,
                                 'Paziente A': round(v1, 3),
                                 'Paziente B': round(v2, 3),
                                 'Delta (A−B)': round(v1-v2, 3)})
            if rows:
                df_cmp = pd.DataFrame(rows)
                st.dataframe(
                    df_cmp.style.background_gradient(
                        subset=['Paziente A','Paziente B'],
                        cmap='RdYlGn_r', vmin=0, vmax=1
                    ),
                    use_container_width=True
                )


# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    RiskScore DM2 Rimini 2026 &nbsp;|&nbsp; AUSL Romagna &nbsp;|&nbsp;
    n=12.526 pazienti DM2 &nbsp;|&nbsp;
    AUC mortalità 0.821 (XGBoost) &nbsp;|&nbsp;
    <b>Uso esclusivo per supporto clinico interno.
    Non sostituisce il giudizio del medico.</b>
</div>
""", unsafe_allow_html=True)
