import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(
    page_title="RiskScore DM2 Rimini",
    page_icon="🏥",
    layout="wide"
)

def score_punti(hba1c, terapia_ipo, dcsi, durata_dm):
    pts  = 0 if hba1c <= 56 else (2 if hba1c <= 75 else 4)
    pts += 4 if terapia_ipo else 0
    pts += 0 if dcsi == 0 else (2 if dcsi == 1 else 4)
    pts += 0 if durata_dm <= 2 else (2 if durata_dm <= 10 else 4)
    return pts

def categoria(sp):
    if sp <= 5:  return "🟢 BASSO",  "green",  "MMG diretta"
    if sp <= 10: return "🟡 MEDIO",  "orange", "MMG + follow-up 6 mesi"
    return             "🔴 ALTO",   "red",    "Diabetologo specialista"

st.sidebar.header("📋 Dati Paziente")
hba1c      = st.sidebar.slider("HbA1c (mmol/mol)", 30, 130, 68)
dcsi       = st.sidebar.number_input("DCSI basale (0-8+)", 0, 12, 1)
terapia_ipo= st.sidebar.checkbox("Terapia a rischio ipoglicemia (ins./SU)")
durata     = st.sidebar.number_input("Durata DM (anni)", 0, 50, 8)
eta        = st.sidebar.number_input("Età (anni)", 18, 100, 65)
egfr       = st.sidebar.slider("eGFR (ml/min)", 10, 150, 72)
drug_sc    = st.sidebar.slider("DrugScore basale", 0, 60, 22)

sp = score_punti(hba1c, terapia_ipo, dcsi, durata)
cat, color, azione = categoria(sp)

st.title("🏥 RiskScore DM2 — Rimini 2026")
st.caption("Registro Diabetologico AUSL Romagna | n=12.526 | Validato 2026")

col1, col2, col3 = st.columns(3)
col1.metric("Score a Punti", f"{sp} / 16")
col2.metric("Categoria",     cat)
col3.metric("Azione",        azione)

st.divider()

# Gauge
fig, ax = plt.subplots(figsize=(10, 2.5))
ax.set_xlim(0, 16); ax.set_ylim(0, 1); ax.set_yticks([])
ax.axvspan(0,  5.5, alpha=0.3, color="green")
ax.axvspan(5.5,10.5, alpha=0.3, color="orange")
ax.axvspan(10.5, 16, alpha=0.3, color="red")
ax.axvline(sp, color="black", lw=4)
ax.scatter([sp], [0.5], s=500, c="black", zorder=5)
ax.text(sp, 0.87, str(sp), ha="center", fontsize=16, fontweight="bold")
for xc, label in [(2.75,"BASSO"), (8.0,"MEDIO"), (13.3,"ALTO")]:
    ax.text(xc, 0.5, label, ha="center", va="center",
            fontsize=13, fontweight="bold", color="white")
ax.set_xticks(range(0, 17, 2))
ax.set_xlabel("Score a Punti (0-16)", fontsize=11)
st.pyplot(fig)

st.divider()
st.subheader("🔍 Dettaglio Punti")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("HbA1c", f"{0 if hba1c<=56 else (2 if hba1c<=75 else 4)} pt")
col_b.metric("TerapiaIpo", f"{4 if terapia_ipo else 0} pt")
col_c.metric("DCSI",       f"{0 if dcsi==0 else (2 if dcsi==1 else 4)} pt")
col_d.metric("DurataDM",   f"{0 if durata<=2 else (2 if durata<=10 else 4)} pt")

st.divider()
st.subheader("📋 Raccomandazione Clinica")
if sp <= 5:
    st.success("Paziente eleggibile per gestione MMG. Follow-up annuale.")
elif sp <= 10:
    st.warning("Valutazione diabetologica consigliata entro 6 mesi. Follow-up semestrale.")
else:
    st.error("Invio a Diabetologo Specialista. Screening complicanze completo.")

st.caption("Per uso clinico interno — Non sostituisce il giudizio del medico")
