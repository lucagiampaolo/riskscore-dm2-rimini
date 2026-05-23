# RiskScore DM2 — Rimini 2026

Score predittivo multi-outcome per pazienti con DM2.  
Registro Diabetologico AUSL Romagna | n=12.526 pazienti | Validato 2026

## Struttura Repository

```
├── app.py                  # App Streamlit principale
├── params_v2.pkl           # Parametri preprocessing (imputer + scaler)
├── mtl_v2_best.h5          # Modello DNN multi-task (Git LFS)
├── XGB_models/             # Modelli XGBoost per ogni outcome
│   ├── xgb_Decesso_finefu.json
│   └── ...
├── requirements.txt
└── README.md
```

## Installazione

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Feature input (10 variabili)

| Feature | Descrizione |
|---------|-------------|
| EtaBasale | Età anagrafica (anni) |
| HbA1cBasale | HbA1c basale (mmol/mol) |
| eGFR_basale | GFR stimato (ml/min/1.73m²) |
| Albuminuria_basale | 0=normo, 1=micro, 2=macro |
| DrugScore_basale | Complessità farmacologica |
| TerapiaRischioIpo | Insulina/SU (0/1) |
| Insulina_multiniettiva_basale | Insulina iniettiva (0/1) |
| DurataDM | Durata diabete (anni) |
| TOD_ER_max_basale | N. organi bersaglio colpiti (0/1/2) |
| ASCVD_basale | Evento CV accertato (0/1) |

## Score disponibili

- **Score a Punti (4V)**: formula semplice 0-16 (carta e penna)
- **Score DNN (7 domini)**: rete neurale multi-task
- **Score XGBoost**: un modello per ogni outcome (~90 outcome)

## Performance validata

| Outcome | Metodo | AUC |
|---------|--------|-----|
| Mortalità (finefu) | XGBoost | **0.821** |
| Percorso MMG | XGBoost | 0.765 |
| MACE5 | XGBoost | 0.758 |
| DCSI_3a (ρ) | XGBoost | 0.704 |

## ⚠️ Uso

Strumento di supporto clinico interno AUSL Romagna.  
Non sostituisce il giudizio clinico del medico.  
Non per uso diagnostico autonomo.
