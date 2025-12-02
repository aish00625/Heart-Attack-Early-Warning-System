# app.py — Final clean version: fixed navbar, medium (≈550px) centered charts, full descriptions
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
import shap

# initialize SHAP JS (safe even if not interactive)
shap.initjs()

# ---------------- Page config ----------------
st.set_page_config(layout="wide", page_title="Heart Attack Dashboard")

# ---------------- Visual constants ----------------
FIG_INCHES = (5, 3.5)  # ~500x350 px depending on DPI
FIG_DPI = 100
PLOT_FONT = 11
PLOT_PX = 550  # target pixel width for plot container

sns.set_theme(style="whitegrid", rc={
    "axes.titlesize": PLOT_FONT + 1,
    "axes.labelsize": PLOT_FONT,
    "xtick.labelsize": PLOT_FONT - 1,
    "ytick.labelsize": PLOT_FONT - 1
})

# ---------------- CSS (navbar + plot limits) ----------------
st.markdown(
    f"""
    <style>
    html, body {{ scroll-behavior: smooth; }}

    /* Fixed navbar */
    .fixed-top-nav {{
      position: fixed !important;
      top: 0;
      left: 0;
      right: 0;
      height: 56px;
      background: linear-gradient(90deg,#0b5fff 0%,#06c2ac 100%);
      color: white;
      padding: 10px 18px;
      display:flex;
      gap:14px;
      align-items:center;
      z-index:2147483647;
      box-shadow: 0 6px 20px rgba(11,95,255,0.12);
      font-weight:700;
      font-family: "Segoe UI", Roboto, Arial, sans-serif;
    }}
    .fixed-top-nav a {{
      color: white !important;
      text-decoration: none;
      padding: 6px 10px;
      border-radius: 8px;
      font-size:14px;
    }}
    .fixed-top-nav a:hover {{
      background: rgba(255,255,255,0.12);
    }}

    /* spacer so content starts below navbar */
    .blocker-space {{
      padding-top: 72px;
    }}

    /* center container for medium charts */
    .plot-container {{
      width: {PLOT_PX}px;
      max-width: 95%;
      margin-left: auto;
      margin-right: auto;
    }}

    /* ensure Streamlit pyplot/plotly respect the medium width */
    .element-container div[data-testid="stPyplot"] img,
    .element-container div[data-testid="stPyplot"] canvas,
    .element-container div[data-testid="stPlotlyChart"] img,
    .element-container div[data-testid="stPlotlyChart"] canvas {{
        width: {PLOT_PX}px !important;
        max-width: 95% !important;
        height: auto !important;
        margin-left: auto !important;
        margin-right: auto !important;
        display: block !important;
    }}

    /* small info card */
    .info-card {{
      background: #ffffffcc;
      border-radius: 8px;
      padding: 8px;
      box-shadow: 0 3px 10px rgba(0,0,0,0.06);
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Smooth scroll JS ----------------
st.components.v1.html(
    """
    <script>
    document.addEventListener("click", function(e){
      const a = e.target.closest('a');
      if(a && a.getAttribute('href')?.startsWith('#')) {
        e.preventDefault();
        const id = a.getAttribute('href').slice(1);
        const el = document.getElementById(id);
        if(el) window.scrollTo({ top: el.offsetTop - 70, behavior: 'smooth' });
      }
    });
    </script>
    """,
    height=0,
    width=0
)

# ---------------- Navbar HTML ----------------
nav_html = """
<div class="fixed-top-nav">
  <div style="font-weight:900;margin-right:10px;font-size:15px">❤️ Heart Attack Dashboard</div>
  <a href="#dataset">Dataset</a>
  <a href="#eda">EDA</a>
  <a href="#correlation">Correlation</a>
  <a href="#model">Model</a>
  <a href="#shap">SHAP</a>
  <a href="#predict">Predict</a>
  <div style="flex:1"></div>
</div>
"""
st.markdown(nav_html, unsafe_allow_html=True)
st.markdown('<div class="blocker-space"></div>', unsafe_allow_html=True)

# ---------------- Helpers ----------------
def prepare_fig(figsize=FIG_INCHES, dpi=FIG_DPI):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.set_dpi(dpi)
    fig.set_size_inches(figsize[0], figsize[1], forward=True)
    plt.rcParams.update({'font.size': PLOT_FONT})
    return fig, ax

@st.cache_data
def load_data(path="heart.csv"):
    return pd.read_csv(path)

@st.cache_data
def train_model(df, target_col="output"):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    explainer = shap.Explainer(rf, X_train)
    return {
        "model": rf,
        "explainer": explainer,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": X.columns.tolist()
    }

# ---------------- Load & Train ----------------
df = load_data("heart.csv")
m = train_model(df, target_col="output")
rf = m["model"]
explainer = m["explainer"]
X_train = m["X_train"]
X_test = m["X_test"]
y_test = m["y_test"]
feature_names = m["feature_names"]

# ---------------- Page header ----------------
st.title("Heart Attack Early Warning — Dashboard")
st.divider()

# ---------------- Dataset ----------------
st.markdown('<a id="dataset"></a>', unsafe_allow_html=True)
st.header("Dataset")
with st.container():
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.dataframe(df.head(12), height=220)
    st.caption("Dataset preview — 'output' is the target (0 = no disease, 1 = disease).")
    st.markdown('</div>', unsafe_allow_html=True)

with st.expander("Dataset — What it shows, how to read, interpretation, next steps"):
    st.write(
        "What it shows\n"
        "- A table with the first 12 rows of the dataset and all feature column names. "
        "The target column is `output` (0 = no disease, 1 = disease).\n\n"
        "How to read it\n"
        "- Each row is one patient record.\n"
        "- Columns are clinical features (age, sex, cp, trtbps, chol, etc.).\n"
        "- `output` is the label used for model training and evaluation.\n\n"
        "What to watch for / interpretation\n"
        "- Missing values or obviously wrong entries (e.g., negative ages).\n"
        "- Unexpected datatypes (strings in numeric columns).\n"
        "- Value ranges and typical examples (e.g., typical blood pressure and cholesterol ranges).\n\n"
        "Suggested next steps\n"
        "- Report counts of missing values and perform appropriate imputations if needed.\n"
        "- Standardize/encode categorical columns before modeling (if any).\n"
        "- Save a cleaned copy or document any manual corrections."
    )

st.divider()

# ---------------- EDA ----------------
st.markdown('<a id="eda"></a>', unsafe_allow_html=True)
st.header("Exploratory Data Analysis (EDA)")

# Age distribution
with st.container():
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    fig, ax = prepare_fig()
    sns.histplot(df['age'], kde=True, ax=ax)
    ax.set_title("Age distribution")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    st.caption("Shows which ages are most common in the dataset.")
    st.markdown('</div>', unsafe_allow_html=True)

with st.expander("Age distribution — What it shows, how to read, interpretation, next steps"):
    st.write(
        "What it shows\n"
        "- A histogram (with KDE) of patient ages in the dataset showing frequency of each age band.\n\n"
        "How to read it\n"
        "- The x-axis is age; the y-axis is count (how many patients).\n"
        "- The KDE (smoothed line) shows the shape of the distribution.\n"
        "- Peaks indicate age groups with the most data.\n\n"
        "What to watch for / interpretation\n"
        "- If distribution is concentrated (e.g., mostly middle-aged), model predictions for underrepresented ages (very young/very old) may be unreliable.\n"
        "- Multiple peaks may indicate mixed cohorts.\n\n"
        "Suggested next steps\n"
        "- Stratify model evaluation by age bands to check fairness.\n"
        "- Consider binning age for models that benefit from categorical bins."
    )

# Cholesterol distribution
with st.container():
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    fig, ax = prepare_fig()
    sns.histplot(df['chol'], kde=True, ax=ax)
    ax.set_title("Cholesterol distribution")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    st.caption("Cholesterol distribution highlights potential outliers.")
    st.markdown('</div>', unsafe_allow_html=True)

with st.expander("Cholesterol distribution — What it shows, how to read, interpretation, next steps"):
    st.write(
        "What it shows\n"
        "- A histogram and KDE for serum cholesterol values showing the spread and presence of outliers.\n\n"
        "How to read it\n"
        "- x-axis is cholesterol (mg/dl); y-axis is count.\n"
        "- The KDE reveals the smooth pattern; a long right tail indicates high-value outliers.\n\n"
        "What to watch for / interpretation\n"
        "- Heavy right tail / outliers: a few patients with very high cholesterol. These can skew mean-based statistics and some models.\n\n"
        "Suggested next steps\n"
        "- Consider winsorizing or log-transforming extreme cholesterol values if they distort model training.\n"
        "- Run sensitivity tests with and without extremes."
    )

st.divider()

# ---------------- Correlation ----------------
st.markdown('<a id="correlation"></a>', unsafe_allow_html=True)
st.header("Correlation Heatmap (medium)")
with st.container():
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = prepare_fig()
    sns.heatmap(corr, annot=True, fmt=".2f", mask=mask, cmap="vlag",
                linewidths=0.4, cbar_kws={"shrink":0.6}, ax=ax, annot_kws={"fontsize":10})
    ax.set_title("Feature Correlation (upper triangle)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    st.caption("Pairwise correlations; watch for high correlation indicating redundancy.")
    st.markdown('</div>', unsafe_allow_html=True)

with st.expander("Correlation heatmap — What it shows, how to read, interpretation, next steps"):
    st.write(
        "What it shows\n"
        "- Pairwise Pearson correlations between numerical features, with the upper triangle masked.\n\n"
        "How to read it\n"
        "- Each cell is the correlation coefficient (−1 to +1) between two features.\n"
        "- Positive values → features increase together; negative → one increases while the other decreases.\n\n"
        "What to watch for / interpretation\n"
        "- |r| > 0.7 suggests strong correlation and potential multicollinearity.\n\n"
        "Suggested next steps\n"
        "- Remove or combine highly correlated features if necessary; consider PCA or feature selection."
    )

st.divider()

# ---------------- Model Performance ----------------
st.markdown('<a id="model"></a>', unsafe_allow_html=True)
st.header("Model Performance (Random Forest)")

y_prob = rf.predict_proba(X_test)[:, 1]
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

col1, col2 = st.columns([1, 2], gap="large")
with col1:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.metric("Accuracy (test)", f"{acc:.3f}")
    st.metric("ROC-AUC (test)", f"{auc:.3f}")
    st.caption("AUC is preferred for imbalanced or clinical classification.")
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    fig, ax = prepare_fig()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    st.caption("Confusion matrix — focus on minimizing False Negatives (missed disease).")
    st.markdown('</div>', unsafe_allow_html=True)

with st.expander("Confusion matrix — What it shows, how to read, interpretation, next steps"):
    st.write(
        "What it shows\n"
        "- Counts of true negatives (TN), false positives (FP), false negatives (FN), and true positives (TP).\n\n"
        "How to read it\n"
        "- Rows = actual class, columns = predicted class.\n\n"
        "What to watch for / interpretation\n"
        "- False Negatives (FN) are critical in a heart-attack warning system: missed cases are dangerous.\n\n"
        "Suggested next steps\n"
        "- Tune threshold to reduce FN (increase sensitivity) and present trade-offs."
    )

# ROC
with st.container():
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    fig, ax = prepare_fig()
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}", color="#0b5fff")
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    st.caption("ROC curve: choose operating threshold to balance FN / FP.")
    st.markdown('</div>', unsafe_allow_html=True)

with st.expander("ROC curve — What it shows, how to read, interpretation, next steps"):
    st.write(
        "What it shows\n"
        "- Model performance across thresholds; plots True Positive Rate vs False Positive Rate.\n\n"
        "How to read it\n"
        "- A curve closer to the top-left indicates better separation.\n\n"
        "Suggested next steps\n"
        "- Choose threshold based on clinical priorities; consider calibration checks."
    )

st.divider()

# ---------------- SHAP ----------------
st.markdown('<a id="shap"></a>', unsafe_allow_html=True)
st.header("SHAP Explainability (Global & Local)")
with st.spinner("Computing SHAP values..."):
    shap_vals = explainer(X_test)

# Global SHAP importance - fallback plotting
with st.container():
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    vals = getattr(shap_vals, "values", None)
    if vals is not None:
        if vals.ndim == 3:
            class_index = 1 if vals.shape[2] > 1 else 0
            vals_for_class = vals[:, :, class_index]
        else:
            vals_for_class = vals
        abs_mean = np.mean(np.abs(vals_for_class), axis=0)
        if len(abs_mean) == len(feature_names):
            imp_df = pd.DataFrame({"feature": feature_names, "importance": abs_mean}).sort_values("importance", ascending=True)
            fig, ax = prepare_fig()
            ax.barh(imp_df['feature'], imp_df['importance'], color="#06c2ac")
            ax.set_title("Mean |SHAP value| per feature")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
            st.caption("Mean |SHAP| indicates average influence of features.")
        else:
            st.write("SHAP fallback mismatch; cannot show bar chart.")
    else:
        st.write("No SHAP values available.")
    st.markdown('</div>', unsafe_allow_html=True)

with st.expander("SHAP global bar — What it shows, how to read, interpretation, next steps"):
    st.write(
        "What it shows\n"
        "- Average absolute SHAP value per feature (for the positive class if appropriate), ranking features by overall influence.\n\n"
        "How to read it\n"
        "- Longer bar = larger mean absolute contribution to predictions.\n\n"
        "Suggested next steps\n"
        "- Validate top features with clinicians; investigate unexpected high-importance features."
    )

# Static beeswarm-like
with st.container():
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    try:
        vals = getattr(shap_vals, "values", None)
        if vals is None:
            raise ValueError("No SHAP values available")
        if vals.ndim == 3:
            class_idx = 1 if vals.shape[2] > 1 else 0
            shap_arr = vals[:, :, class_idx]
        else:
            shap_arr = vals
        mean_abs = np.mean(np.abs(shap_arr), axis=0)
        top_idx = np.argsort(mean_abs)[-10:][::-1]
        top_features = [feature_names[i] for i in top_idx]
        rows = []
        X_test_r = X_test.reset_index(drop=True)
        for i_feat in top_idx:
            feat_shap = shap_arr[:, i_feat]
            feat_name = feature_names[i_feat]
            feat_vals = X_test_r[feat_name].values if feat_name in X_test_r.columns else np.zeros_like(feat_shap)
            for s_val, f_val in zip(feat_shap, feat_vals):
                rows.append({"feature": feat_name, "shap_value": s_val, "feat_value": f_val})
        plot_df = pd.DataFrame(rows)
        fig, ax = prepare_fig()
        sns.stripplot(x="shap_value", y="feature", data=plot_df, order=top_features,
                      jitter=0.35, size=4, ax=ax)
        ax.axvline(0, color="#333333", linestyle="--", linewidth=0.8)
        ax.set_xlabel("SHAP value (impact)")
        ax.set_ylabel("")
        ax.set_title("Static SHAP beeswarm-like (top features)")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        st.caption("Static beeswarm-like plot: each dot = sample SHAP value; right increases predicted risk.")
    except Exception as e:
        st.write("Could not draw beeswarm-like plot:", e)
    st.markdown('</div>', unsafe_allow_html=True)

with st.expander("SHAP beeswarm — What it shows, how to read, interpretation, next steps"):
    st.write(
        "What it shows\n"
        "- Per-sample SHAP values for top features visualized as jittered dots. Horizontal position = SHAP value.\n\n"
        "How to read it\n"
        "- Dots to the right increase predicted risk; dots to the left decrease it.\n\n"
        "Suggested next steps\n"
        "- Use local SHAP to explain individual predictions and investigate interactions."
    )

st.divider()

# ---------------- Predict & Explain ----------------
st.markdown('<a id="predict"></a>', unsafe_allow_html=True)
st.header("Predict & Explain (Single Patient)")
with st.form("patient_form_full", clear_on_submit=False):
    cols = st.columns(3)
    age = cols[0].number_input("age", min_value=1, max_value=120, value=int(df['age'].median()))
    sex = cols[1].selectbox("sex (1=male,0=female)", options=[1, 0], index=0)
    cp = cols[2].number_input("cp (chest pain type, 0-3)", min_value=0, max_value=3, value=int(df['cp'].mode()[0]))
    trtbps = st.number_input("resting blood pressure (trtbps)", value=int(df['trtbps'].median()))
    chol = st.number_input("cholesterol (chol)", value=int(df['chol'].median()))
    fbs = st.selectbox("fasting blood sugar >120 mg/dl (fbs)", options=[1, 0], index=0)
    restecg = st.number_input("restecg (0-2)", min_value=0, max_value=2, value=int(df['restecg'].mode()[0]))
    thalachh = st.number_input("max heart rate achieved (thalachh)", value=int(df['thalachh'].median()))
    exng = st.selectbox("exercise induced angina (exng)", options=[1, 0], index=0)
    oldpeak = st.number_input("oldpeak (ST depression)", value=float(df['oldpeak'].median()))
    slp = st.number_input("slope (slp) 0-2", min_value=0, max_value=2, value=int(df['slp'].mode()[0]))
    caa = st.number_input("number of major vessels (caa) 0-3", min_value=0, max_value=3, value=int(df['caa'].mode()[0]))
    thall = st.number_input("thal (thall) 0-3", min_value=0, max_value=3, value=int(df['thall'].mode()[0]))
    submitted = st.form_submit_button("Predict & Explain")

if submitted:
    sample = pd.DataFrame([{
        'age': age, 'sex': sex, 'cp': cp, 'trtbps': trtbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalachh': thalachh, 'exng': exng,
        'oldpeak': oldpeak, 'slp': slp, 'caa': caa, 'thall': thall
    }])
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.markdown("**Patient input**")
    st.dataframe(sample)
    risk = rf.predict_proba(sample)[:, 1][0]
    st.metric("Predicted Heart Attack Risk", f"{risk*100:.2f}%")
    st.caption("Model probability of heart disease for this patient.")

    # local SHAP
    s_expl = explainer(sample)
    vals = getattr(s_expl, "values", None)
    if vals is not None:
        if vals.ndim == 3:
            local_shap = vals[0, :, 1] if vals.shape[2] > 1 else vals[0, :, 0]
        else:
            local_shap = vals[0]
    else:
        local_shap = np.zeros(len(feature_names))

    local_df = pd.DataFrame({"feature": feature_names, "shap_value": local_shap})
    local_df = local_df.reindex(local_df['shap_value'].abs().sort_values(ascending=False).index).head(10)
    fig, ax = prepare_fig()
    ax.barh(local_df['feature'], local_df['shap_value'], color="#0b5fff")
    ax.set_xlabel("SHAP value (impact)")
    ax.set_title("Top local feature contributions (SHAP)")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    st.caption("Positive SHAP increases risk; negative SHAP decreases risk.")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Local SHAP — What it shows, how to read, interpretation, next steps"):
        st.write(
            "What it shows\n"
            "- For a single patient input, the bar chart shows the SHAP contribution of each feature to the predicted risk.\n\n"
            "How to read it\n"
            "- Bars sorted by absolute impact; positive bars increase predicted risk.\n\n"
            "Suggested next steps\n"
            "- Use local SHAP to explain individual predictions; review surprising explanations."
        )

st.divider()