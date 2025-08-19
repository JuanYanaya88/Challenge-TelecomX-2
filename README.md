# Challenge-TelecomX-2

Challenge Telecom X: análisis de evasión de clientes - Parte 2
---

## 🧠 1) Resumen 

**Objetivo:** Anticipar la cancelación de clientes (churn) para activar estrategias de retención con mayor retorno sobre la inversión (ROI).

**Enfoque:**  
Se construyó un pipeline de machine learning con:
- Preparación de datos estructurada
- Selección de variables dentro del pipeline (evitando fugas)
- Comparación entre modelos: **Regresión Logística con penalización L1 (LR_L1_select)** vs **Random Forest (RF_select)**

**Resultados clave (holdout 30%):**

| Modelo         | ROC-AUC | PR-AUC | Accuracy | Recall (churn) | Precisión (churn) | F1   |
|----------------|--------:|-------:|---------:|----------------:|-------------------:|-----:|
| **LR_L1_select** | **0.845** | **0.659** | 0.75     | **0.81**         | 0.52               | **0.63** |
| **RF_select**    | 0.824   | 0.609  | **0.78** | 0.60             | **0.60**           | 0.59  |

**Interpretación ejecutiva:**
- **LR** prioriza capturar churners (recall alto) → útil cuando perder un cliente es costoso.
- **RF** reduce falsas alarmas (mayor precisión y accuracy) → ideal cuando el presupuesto de retención es limitado.
- **Umbral operativo sugerido (LR):** ~0.505 (maximiza F1). Ajustable según estrategia comercial.

---

## 🗂️ 2) Estructura del Proyecto

```bash
.
├─ notebooks/
│  └─ latam_final_2.ipynb        # Cuaderno principal (EDA + modelado)
├─ data/
│  ├─ raw/                       # Datos crudos (opcional)
│  └─ processed/
│     └─ df_limpo.csv            # Datos tratados
├─ reports/
│  └─ figures/                   # Visualizaciones (EDA, coeficientes, importancias)
├─ models/                       # Artefactos entrenados (opcional)
└─ tabla_modelos_cv_holdout.csv  # Comparativa de métricas (opcional)
```

---

## 🧼 3) Preparación de Datos

- **Categóricas → dummies** (`drop_first=True`): contrato, método de pago, servicios (internet, seguridad, soporte).
- **Numéricas:** tenure, cargos mensuales y totales.
- **Normalización:** `StandardScaler` (solo para LR).
- **Desbalanceo:** `SMOTE` aplicado dentro del pipeline (solo en entrenamiento).
- **Split:** 70% entrenamiento / 30% test estratificado + validación cruzada 5-fold.
- **Selección de variables (sin fugas):**  
  - `SelectFromModel` con **LR L1**  
  - `SelectFromModel` con **RF** (`threshold='median'`)

**Decisiones clave:**
- Dummies y escalado optimizan modelos lineales.
- SMOTE y selección dentro de CV evitan fugas y mejoran recall.
- Métricas orientadas a desbalance (PR-AUC, recall) + ranking global (ROC-AUC).
- Umbral ajustable según sensibilidad comercial (recall vs precisión).

---

## 🔍 4) EDA — Insights Relevantes

- Mayor riesgo de churn en:
  - Contrato **month-to-month**
  - **Tenure bajo**
  - **Cargos altos**
  - Método de pago: **Electronic check**
  - Ausencia de **OnlineSecurity** y **TechSupport**

- Factores protectores:
  - Contratos de **1 o 2 años**
  - **Tenure alto**
  - Servicios activos de seguridad y soporte

**Visualizaciones clave:**
- Heatmap de correlaciones
- Distribuciones por churn
- Barras por categoría
- Coeficientes (LR) e Importancias (RF)

---

## 📈 5) Factores que Más Influyen en el Churn

Confirmados por coeficientes LR y/o importancias RF:

| Factor                     | Impacto en Churn |
|---------------------------|------------------|
| Contrato month-to-month   | ↑ Churn          |
| Contrato 1/2 años         | ↓ Churn          |
| Tenure bajo               | ↑ Churn          |
| Tenure alto               | ↓ Churn          |
| Cargos altos              | ↑ Churn          |
| Electronic check          | ↑ Churn          |
| Sin OnlineSecurity/TechSupport | ↑ Churn    |
| InternetService (Fiber optic) | Relevante (ver signo) |

---

## 🛠️ 6) Recomendaciones de Retención

### 📊 Segmentación por riesgo (probabilidad de churn, p)

| Riesgo | Perfil típico | Acciones recomendadas |
|--------|---------------|------------------------|
| Alto (p ≥ 0.60) | Contrato mensual, tenure < 12m, cargos altos, sin seguridad/soporte, Electronic check | Permanencia 12/24m con descuento, bundle de seguridad/soporte por 3–6m |
| Medio (0.40 ≤ p < 0.60) | Clientes indecisos | Prueba de add-ons, migración a contrato anual con incentivo, educación de valor |
| Bajo (p < 0.40) | Clientes fidelizados | Comunicaciones de mantenimiento, programa de referidos |

### 🧪 Experimentos A/B sugeridos

- Comparar **Oferta A (descuento)** vs **Oferta B (bundle)** en clientes de alto riesgo
- Mensajes: ahorro económico vs tranquilidad (seguridad/soporte)
- Métrica principal: retención a 60–90 días y uplift vs grupo control

---

## ⚙️ 7) Ejecución Rápida

### 📦 Requisitos

```bash
pip install -U pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib statsmodels
```

### 📥 Cargar datos

```python
import pandas as pd
df = pd.read_csv('data/processed/df_limpo.csv')  # En Colab: '/content/telecomx2.csv'
```
---
