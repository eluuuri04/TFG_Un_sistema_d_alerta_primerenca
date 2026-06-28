"""
Sistema d'Alerta Primerenca — Fonaments de Física
Eina de seguiment per identificar, amb antelació, els estudiants amb risc
de no superar l'assignatura, a partir de la seva activitat a l'Aula Virtual
i el seu rendiment acadèmic parcial.

Execució local:  streamlit run app.py
"""

import json
import os
import re

import lightgbm as lgb
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────
# Configuració general
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EWS · Fonaments de Física",
    page_icon="🎓",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "resultats")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Moments de seguiment al llarg del curs
CHECKPOINTS = [
    {"key": "b1", "label": "Bloc 1", "date": "fins 11 abril", "short": "B1"},
    {"key": "b2", "label": "Bloc 2", "date": "fins 12 maig", "short": "B2"},
    {"key": "b3", "label": "Bloc 3", "date": "fins 27 maig", "short": "B3"},
    {"key": "b4", "label": "Bloc 4", "date": "curs complet", "short": "B4"},
]

DROP_BLOCK = ["target", "id", "grup", "semi", "lab", "aval", "total_examen"]
DROP_GENERAL = ["target", "id", "grup", "semi", "lab", "aval"]

AVAL_LABELS = {
    "AvAlt_ExamOnly": "Examen final únic",
    "AvAlt_withAC": "Contínua",
}

# Patrons de noms de columna que corresponen a notes/qualificacions ja obtingudes
# (no es poden canviar a posteriori), per distingir-les de l'activitat a l'Aula
# Virtual (que sí que es pot reforçar de cara endavant).
ACADEMIC_PATTERNS = [
    r"_final$", r"_q_total$", r"_pdf_total$", r"^total_p\d", r"^total_s\d",
    r"^repas_b", r"mitjana", r"^total_examen$", r"^ex_recup$", r"^ex_final$",
    r"^nota_final$", r"^b\d+_(prac|sem)$", r"^s\d+_no_aval$", r"^s\d+_opt$",
    r"^p_global$", r"^s_global$", r"^total_practiques$", r"^total_seminaris$",
    r"^total_quizzes$",
]


def is_academic_feature(original_name):
    return any(re.search(p, original_name) for p in ACADEMIC_PATTERNS)

PREFIXES_TO_STRIP = [
    "Qu_estionari_", "Qu_esitonari_", "Quiz__", "Quiz_",
    "Fitxer__", "Fitxer_", "File__", "File_",
    "Course_Root_", "Course_root_",
    "Exercicis_Sessi_", "Teoria_Sessi_",
]


def shorten(name, max_len=38):
    for p in PREFIXES_TO_STRIP:
        if name.startswith(p):
            name = name[len(p):]
            break
    if len(name) > max_len:
        name = name[: max_len - 1] + "…"
    return name


def sanitize_cols(cols):
    return [re.sub(r"[^A-Za-z0-9_]+", "_", c) for c in cols]


# ──────────────────────────────────────────────────────────────────────────
# Càrrega de dades i models (amb cache)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Carregant el sistema...")
def load_models():
    models = {}
    for cp in CHECKPOINTS:
        models[cp["key"]] = lgb.Booster(
            model_file=os.path.join(MODELS_DIR, f"model_{cp['key']}_optuna.txt"))
    models["general"] = lgb.Booster(model_file=os.path.join(MODELS_DIR, "model_general.txt"))
    with open(os.path.join(MODELS_DIR, "thresholds_optuna.json")) as f:
        thresholds = json.load(f)
    return models, thresholds


@st.cache_data(show_spinner="Carregant dades dels estudiants...")
def load_data():
    data = {cp["key"]: pd.read_csv(os.path.join(DATA_DIR, f"{cp['key']}.csv")) for cp in CHECKPOINTS}
    data["qualificacions"] = pd.read_csv(os.path.join(DATA_DIR, "qualificacions.csv"))
    data["alt_df"] = pd.read_csv(os.path.join(DATA_DIR, "alt_df.csv"))
    return data


@st.cache_resource(show_spinner=False)
def get_explainer(key, _model):
    return shap.TreeExplainer(_model)


def col_mapping(original_cols):
    """Sanitized column name -> nom original."""
    return dict(zip(sanitize_cols(original_cols), original_cols))


@st.cache_resource(show_spinner="Calculant l'explicabilitat del grup...")
def compute_shap_matrix(block_key, _model, _df):
    """SHAP values per a tots els estudiants d'un bloc. Retorna (sv, X, original_cols)."""
    X, original_cols = get_features(_df, block_key)
    explainer = shap.TreeExplainer(_model)
    sv = explainer.shap_values(X)
    if sv.ndim == 3:
        sv = sv[:, :, -1] if sv.shape[-1] == 2 else sv[..., 0]
    return sv, X, original_cols


def get_features(df, block_key):
    """Prepara les dades de l'estudiant tal com les espera el sistema."""
    drop_cols = DROP_GENERAL if block_key == "general" else DROP_BLOCK
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    original_cols = list(X.columns)
    X.columns = sanitize_cols(X.columns)
    return X, original_cols


def predict_all(model, df, block_key):
    X, _ = get_features(df, block_key)
    return model.predict(X)


models, thresholds = load_models()
data = load_data()


def threshold_for(block_key):
    return thresholds[block_key]


# ──────────────────────────────────────────────────────────────────────────
# Sidebar — Navegació
# ──────────────────────────────────────────────────────────────────────────
st.sidebar.title("🎓 Sistema d'Alerta Primerenca")
st.sidebar.caption("Fonaments de Física · Seguiment dels estudiants")
page = st.sidebar.radio(
    "Navegació",
    ["🏠 Panell de seguiment", "🔍 Fitxa de l'estudiant", "📌 Factors de risc", "❓ Com funciona"],
)
st.sidebar.divider()
cp_labels = [cp["label"] for cp in CHECKPOINTS]
moment_label = st.sidebar.selectbox("Bloc del curs", cp_labels, index=min(1, len(cp_labels) - 1))
cp_sel = next(cp for cp in CHECKPOINTS if cp["label"] == moment_label)
bk_current = cp_sel["key"]

has_grup = "grup" in data["b4"].columns
grup_values_global = sorted(data["b4"]["grup"].dropna().astype(str).unique().tolist()) if has_grup else []
grup_choice = st.sidebar.selectbox(
    "Grup de teoria", ["Tots els grups"] + grup_values_global
) if has_grup else "Tots els grups"


def filter_grup(df):
    if grup_choice == "Tots els grups" or "grup" not in df.columns:
        return df
    return df[df["grup"].astype(str) == grup_choice]


st.sidebar.divider()
st.sidebar.caption("DEMO Desenvolupada amb Intel·ligència Artificial. Dades de l'Aula Virtual i del rendiment acadèmic del curs, anonimitzades.")

# ──────────────────────────────────────────────────────────────────────────
# PÀGINA 1 — Panell de seguiment
# ──────────────────────────────────────────────────────────────────────────
if page == "🏠 Panell de seguiment":
    st.title("Panell de seguiment")
    grup_suffix = f" · Grup {grup_choice}" if grup_choice != "Tots els grups" else ""
    st.caption(f"Estat dels estudiants — **{cp_sel['label']}** ({cp_sel['date']}){grup_suffix}")

    df_cur_all = data[bk_current]
    model_cur = models[bk_current]
    thr_cur = threshold_for(bk_current)

    df_cur = filter_grup(df_cur_all)
    probs = predict_all(model_cur, df_cur, bk_current) if len(df_cur) else pd.Series([], dtype=float)
    en_risc = probs <= thr_cur if len(df_cur) else pd.Series([], dtype=bool)

    total = len(df_cur)
    n_risc = int(en_risc.sum())
    pct_risc = n_risc / total if total else 0

    # Comptatge del bloc anterior (mateix grup), per calcular la tendència
    cp_idx = next(i for i, cp in enumerate(CHECKPOINTS) if cp["key"] == bk_current)
    delta_risc, delta_pct = None, None
    if cp_idx > 0:
        prev_key = CHECKPOINTS[cp_idx - 1]["key"]
        prev_df = filter_grup(data[prev_key])
        if len(prev_df):
            prev_probs = predict_all(models[prev_key], prev_df, prev_key)
            prev_n_risc = int((prev_probs <= threshold_for(prev_key)).sum())
            prev_total = len(prev_df)
            delta_risc = n_risc - prev_n_risc
            delta_pct = pct_risc - (prev_n_risc / prev_total if prev_total else 0)

    # Nivells de risc: separem els estudiants en risc en "Alt" i "Moderat" pel valor de la mediana
    risc_probs = probs[en_risc] if total else probs
    mediana_risc = float(pd.Series(risc_probs).median()) if len(risc_probs) else None
    n_alt = int((risc_probs <= mediana_risc).sum()) if mediana_risc is not None else 0
    n_moderat = n_risc - n_alt
    n_sense_risc = total - n_risc

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estudiants avaluats", total)
    c2.metric("Estudiants en risc", n_risc,
               delta=None if delta_risc is None else int(delta_risc),
               delta_color="inverse")
    c3.metric("% en risc", f"{pct_risc:.0%}",
               delta=None if delta_pct is None else f"{delta_pct:+.0%}",
               delta_color="inverse")
    c4.metric("Risc alt", n_alt)

    st.divider()
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.subheader("Distribució de risc")
        if total == 0:
            st.info("Cap estudiant en aquest grup.")
        else:
            fig_donut = go.Figure(go.Pie(
                labels=["Risc alt", "Risc moderat", "Sense risc"],
                values=[n_alt, n_moderat, n_sense_risc],
                hole=0.55,
                marker=dict(colors=["#e31a1c", "#fdae61", "#33a02c"]),
            ))
            fig_donut.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                                     legend=dict(orientation="h", yanchor="bottom", y=-0.15))
            st.plotly_chart(fig_donut, use_container_width=True)

    with col_b:
        if has_grup:
            st.subheader("Risc per grup de teoria")
            probs_all_grup = predict_all(model_cur, df_cur_all, bk_current)
            en_risc_all_grup = probs_all_grup <= thr_cur
            tmp = pd.DataFrame({"grup": df_cur_all["grup"].astype(str).values, "en_risc": en_risc_all_grup})
            agg = tmp.groupby("grup")["en_risc"].mean().sort_values(ascending=True) * 100
            grp_labels = [str(v) for v in agg.index]
            bar_colors = ["#1f78b4" if g == grup_choice else "#cab2d6" for g in agg.index]
            fig_grp = go.Figure(go.Bar(x=agg.values, y=grp_labels,
                                        orientation="h", marker_color=bar_colors))
            # Els grups són una variable categòrica (no numèrica): forcem explícitament
            # un eix de categories perquè Plotly no l'interpreti com una escala numèrica.
            fig_grp.update_layout(
                height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="% en risc",
                yaxis=dict(type="category", categoryorder="array", categoryarray=grp_labels),
            )
            st.plotly_chart(fig_grp, use_container_width=True)

    st.divider()
    st.subheader("Factors de risc clau")
    st.caption(f"Variables amb més pes en la classificació d'aquest bloc — **{cp_sel['label']}**.")
    importance_dash = model_cur.feature_importance(importance_type="gain")
    feat_names_dash = model_cur.feature_name()
    _, original_cols_dash = get_features(df_cur_all, bk_current)
    mapping_dash = col_mapping(original_cols_dash)
    fi_dash = pd.DataFrame({
        "feature": [shorten(mapping_dash.get(f, f)) for f in feat_names_dash],
        "importance": importance_dash,
    }).sort_values("importance", ascending=False).head(6).sort_values("importance")
    fig_fi_dash = go.Figure(go.Bar(
        x=fi_dash["importance"], y=fi_dash["feature"], orientation="h", marker_color="#1f78b4"
    ))
    fig_fi_dash.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Pes en la decisió")
    st.plotly_chart(fig_fi_dash, use_container_width=True)
    st.caption("Anàlisi detallada a la pàgina **📌 Factors de risc**.")

    st.divider()
    st.subheader("Estudiants en risc en aquest moment")

    if total == 0:
        st.info("Cap estudiant en aquest grup en aquest bloc.")
    else:
        risk_table = pd.DataFrame({
            "ID estudiant": df_cur["id"].values,
            "Probabilitat d'aprovar (%)": (probs * 100).round(1),
        })
        risk_table = risk_table[en_risc].sort_values("Probabilitat d'aprovar (%)")

        if risk_table.empty:
            st.success("Cap estudiant classificat en risc en aquest moment.")
        else:
            st.dataframe(
                risk_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Probabilitat d'aprovar (%)": st.column_config.ProgressColumn(
                        "Probabilitat d'aprovar", format="%.1f%%", min_value=0, max_value=100,
                    )
                },
            )
            st.download_button(
                "⬇️ Descarrega la llista (CSV)",
                data=risk_table.to_csv(index=False).encode("utf-8"),
                file_name=f"estudiants_en_risc_{bk_current}_{grup_choice}.csv",
                mime="text/csv",
            )

    st.divider()
    st.subheader("Evolució del nombre d'estudiants en risc")
    counts = []
    for cp in CHECKPOINTS:
        df_cp = filter_grup(data[cp["key"]])
        if len(df_cp):
            p = predict_all(models[cp["key"]], df_cp, cp["key"])
            counts.append(int((p <= threshold_for(cp["key"])).sum()))
        else:
            counts.append(0)
    fig = go.Figure(go.Scatter(
        x=[cp["label"] for cp in CHECKPOINTS], y=counts, mode="lines+markers",
        line=dict(color="#e31a1c", width=3), marker=dict(size=12),
    ))
    fig.update_layout(height=350, yaxis_title="Estudiants en risc")
    st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────
# PÀGINA 2 — Fitxa de l'estudiant
# ──────────────────────────────────────────────────────────────────────────
elif page == "🔍 Fitxa de l'estudiant":
    st.title("Fitxa de l'estudiant")

    # Llista d'estudiants filtrada pel grup seleccionat a la barra lateral, amb
    # l'estat de risc (segons el bloc seleccionat) visible directament al desplegable.
    ids_pool = sorted(filter_grup(data["b4"])["id"].tolist())
    df_status = data[bk_current]
    probs_status = predict_all(models[bk_current], df_status, bk_current)
    status_map = dict(zip(df_status["id"].values, (probs_status <= threshold_for(bk_current))))

    def _sort_key(i):
        if i not in status_map:
            return (2, i)
        return (0 if status_map[i] else 1, i)

    def _format_id(i):
        if i not in status_map:
            return f"⚪  {i}  ·  sense dades en aquest bloc"
        return f"{'🔴' if status_map[i] else '🟢'}  {i}"

    ids_sorted = sorted(ids_pool, key=_sort_key)

    # Mantenir l'estudiant seleccionat en canviar de bloc o de grup.
    #
    # El text de cada opció inclou l'emoji de risc (🔴/🟢), que depèn del bloc
    # seleccionat. Streamlit identifica internament la selecció pel text
    # formatat que es va enviar al navegador, no per l'ID en cru: si en
    # canviar de bloc l'emoji d'un estudiant canvia (passa de risc a no-risc
    # o viceversa), el text antic ja no coincideix amb cap opció nova i
    # Streamlit "perd" la selecció i la torna a la primera de la llista. Amb
    # un filtre de grup passa una cosa similar si l'estudiant queda exclòs
    # temporalment. Per evitar-ho, guardem la preferència real de l'usuari a
    # part i, cada cop que canvia el bloc o el grup, la tornem a fixar
    # explícitament abans de crear el desplegable.
    WIDGET_KEY = "fitxa_sel_id"
    PREF_KEY = "fitxa_student_pref"
    PREV_CTX_KEY = "fitxa_prev_ctx"

    prev_ctx = st.session_state.get(PREV_CTX_KEY)
    cur_ctx = (grup_choice, bk_current)
    ctx_just_changed = prev_ctx is not None and prev_ctx != cur_ctx
    st.session_state[PREV_CTX_KEY] = cur_ctx

    pref = st.session_state.get(PREF_KEY)
    if ctx_just_changed and pref in ids_sorted:
        st.session_state[WIDGET_KEY] = pref

    sel_id = st.selectbox(
        "Estudiant (ID anonimitzat)", ids_sorted, format_func=_format_id, key=WIDGET_KEY
    )

    if not ctx_just_changed:
        st.session_state[PREF_KEY] = sel_id

    st.caption(
        f"🔴 en risc · 🟢 sense risc — segons **{cp_sel['label']}** "
        f"({'tots els grups' if grup_choice == 'Tots els grups' else 'grup ' + grup_choice})"
    )

    # Informació acadèmica registrada: sempre visible a dalt de la fitxa.
    row_b4_sel = data["b4"][data["b4"]["id"] == sel_id]
    qual = data["qualificacions"]
    row_q = qual[qual["id"] == sel_id]
    nota_final = (row_q["nota_final"].values[0]
                  if len(row_q) and pd.notna(row_q["nota_final"].values[0]) else None)
    target_b4 = row_b4_sel["target"] if len(row_b4_sel) else pd.Series([], dtype=float)
    resultat_real = "Aprovat" if len(target_b4) and target_b4.values[0] == 1 else "Suspès"
    grup_raw = row_b4_sel["grup"].values[0] if len(row_b4_sel) and "grup" in row_b4_sel.columns else None
    grup_sel = "—" if grup_raw is None or pd.isna(grup_raw) else str(grup_raw)
    aval_sel_raw = (row_b4_sel["aval"].values[0]
                    if len(row_b4_sel) and "aval" in row_b4_sel.columns else None)
    if aval_sel_raw is None or pd.isna(aval_sel_raw):
        aval_sel = "Contínua"
    else:
        aval_sel = AVAL_LABELS.get(aval_sel_raw, str(aval_sel_raw))

    st.markdown("##### 📋 Informació acadèmica registrada")
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Grup de teoria", grup_sel)
    i2.metric("Mètode d'avaluació", aval_sel)
    i3.metric("Resultat acadèmic registrat", resultat_real)
    i4.metric("Nota final", f"{nota_final:.2f}" if nota_final is not None else "—")
    st.divider()

    row_cur = df_status[df_status["id"] == sel_id]
    tab_resum, tab_evolucio = st.tabs(["📍 Estat actual", "📈 Evolució al llarg del curs"])

    # ── TAB 1: estat actual ────────────────────────────────────────────────
    with tab_resum:
        if len(row_cur) == 0:
            st.warning("Aquest estudiant no té dades disponibles en aquest bloc del curs.")
        else:
            model_cur_page = models[bk_current]
            X_row, original_cols = get_features(row_cur, bk_current)
            prob = float(model_cur_page.predict(X_row)[0])
            thr = threshold_for(bk_current)
            en_risc = prob <= thr

            cA, cB = st.columns([1, 2])
            with cA:
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(prob * 100, 1),
                    number={"suffix": "%"},
                    title={"text": "Probabilitat d'Aprovar"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#1f78b4"},
                        "steps": [
                            {"range": [0, thr * 100], "color": "#fbb4ae"},
                            {"range": [thr * 100, 100], "color": "#ccebc5"},
                        ],
                        "threshold": {"line": {"color": "#e31a1c", "width": 4}, "value": thr * 100},
                    },
                ))
                fig_g.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
                st.plotly_chart(fig_g, use_container_width=True)
                if en_risc:
                    st.error("🔴 Estudiant classificat **EN RISC**")
                else:
                    st.success("🟢 Estudiant classificat sense risc aparent")

            with cB:
                st.markdown("**Per què el sistema fa aquesta valoració**")
                explainer = get_explainer(bk_current, model_cur_page)
                sv_row = explainer.shap_values(X_row)
                sv_row = sv_row[0] if sv_row.ndim == 2 else sv_row
                mapping = dict(zip(X_row.columns, original_cols))
                fi = pd.DataFrame({
                    "feature": [shorten(mapping[c]) for c in X_row.columns],
                    "original": [mapping[c] for c in X_row.columns],
                    "shap": sv_row,
                })
                fi["abs"] = fi["shap"].abs()
                top = fi.sort_values("abs", ascending=False).head(10).sort_values("shap")
                colors = ["#e31a1c" if v < 0 else "#33a02c" for v in top["shap"]]
                fig_sh = go.Figure(go.Bar(
                    x=top["shap"], y=top["feature"], orientation="h", marker_color=colors
                ))
                fig_sh.update_layout(
                    height=280, margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_title="Verd = afavoreix aprovar · Vermell = afavoreix suspendre",
                )
                st.plotly_chart(fig_sh, use_container_width=True)

            st.divider()
            st.markdown("##### 🎯 Factors a reforçar per millorar la seva probabilitat d'aprovar")
            neg = fi[fi["shap"] < 0].sort_values("shap").head(5)
            if neg.empty:
                st.success(
                    "Ara mateix no hi ha variables que estiguin penalitzant de manera "
                    "rellevant la seva probabilitat d'aprovar."
                )
            else:
                st.caption(
                    "Variables que actualment redueixen més la probabilitat d'aprovar d'aquest "
                    "estudiant. No són causes garantides, sinó els aspectes amb més pes negatiu "
                    "segons el model. Les d'activitat a l'Aula Virtual es poden reforçar de cara "
                    "endavant; les que ja són notes obtingudes no es poden canviar, però assenyalen "
                    "punts a repassar o millorar."
                )
                for _, r in neg.iterrows():
                    tag = " · *punt a repassar*" if is_academic_feature(r["original"]) else ""
                    st.markdown(f"- **{r['original']}**{tag}")

    # ── TAB 2: evolució al llarg del curs ───────────────────────────────────
    with tab_evolucio:
        probs_evol, thr_evol, x_labels, available = [], [], [], []
        for cp in CHECKPOINTS:
            bk = cp["key"]
            df_b = data[bk]
            row = df_b[df_b["id"] == sel_id]
            x_labels.append(cp["label"])
            thr_evol.append(threshold_for(bk))
            if len(row) == 0:
                probs_evol.append(None)
                available.append(False)
            else:
                X, _ = get_features(row, bk)
                prob_e = float(models[bk].predict(X)[0])
                probs_evol.append(prob_e)
                available.append(True)

        st.subheader("Evolució de la probabilitat d'aprovar")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_labels, y=probs_evol, mode="lines+markers", name="Prob. d'aprovar",
                                  line=dict(color="#1f78b4", width=3), marker=dict(size=12)))
        fig.add_trace(go.Scatter(x=x_labels, y=thr_evol, mode="lines+markers", name="Llindar de risc",
                                  line=dict(color="#e31a1c", width=2, dash="dash")))
        fig.update_layout(height=420, yaxis_title="Probabilitat d'aprovar", yaxis_range=[0, 1],
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

        if not all(available):
            st.caption("Nota: aquest estudiant no té dades disponibles en algun dels moments del curs.")

        if len(row_cur):
            st.divider()
            st.subheader("Com han evolucionat els seus factors clau")
            st.caption(f"Variables amb més pes en la valoració actual ({cp_sel['label']}).")

            X_row_e, original_cols_e0 = get_features(row_cur, bk_current)
            explainer_cur = get_explainer(bk_current, models[bk_current])
            sv_cur = explainer_cur.shap_values(X_row_e)
            sv_cur = sv_cur[0] if sv_cur.ndim == 2 else sv_cur
            mapping_cur = dict(zip(X_row_e.columns, original_cols_e0))
            fi_cur = pd.DataFrame({
                "original": [mapping_cur[c] for c in X_row_e.columns],
                "shap": sv_cur,
            })
            fi_cur["abs"] = fi_cur["shap"].abs()
            top4_original = fi_cur.sort_values("abs", ascending=False).head(4)["original"].tolist()
            top4_short = [shorten(n) for n in top4_original]

            evo_x, evo_series = [], {name: [] for name in top4_original}
            for cp in CHECKPOINTS:
                bk_e = cp["key"]
                df_e = data[bk_e]
                row_e = df_e[df_e["id"] == sel_id]
                evo_x.append(cp["label"])
                if len(row_e) == 0:
                    for name in top4_original:
                        evo_series[name].append(None)
                    continue
                X_e, original_cols_e = get_features(row_e, bk_e)
                mapping_e_inv = {v: k for k, v in zip(X_e.columns, original_cols_e)}
                explainer_e = get_explainer(bk_e, models[bk_e])
                sv_e = explainer_e.shap_values(X_e)
                sv_e = sv_e[0] if sv_e.ndim == 2 else sv_e
                for name in top4_original:
                    san = mapping_e_inv.get(name)
                    if san is not None and san in X_e.columns:
                        evo_series[name].append(float(sv_e[list(X_e.columns).index(san)]))
                    else:
                        evo_series[name].append(None)

            fig_evo = go.Figure()
            palette = ["#1f78b4", "#e31a1c", "#33a02c", "#ff7f00"]
            for j, name in enumerate(top4_original):
                fig_evo.add_trace(go.Scatter(
                    x=evo_x, y=evo_series[name], mode="lines+markers", name=top4_short[j],
                    line=dict(color=palette[j % len(palette)], width=3), marker=dict(size=9),
                    connectgaps=False,
                ))
            fig_evo.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                                   yaxis_title="Pes en la decisió (SHAP)",
                                   legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig_evo, use_container_width=True)
            st.caption(
                "Un buit en la línia indica que la variable no estava disponible en aquell bloc."
            )

# ──────────────────────────────────────────────────────────────────────────
# PÀGINA 3 — Factors de risc
# ──────────────────────────────────────────────────────────────────────────
elif page == "📌 Factors de risc":
    st.title("Factors de risc")
    st.markdown(
        f"Variables que més pesen en la classificació de risc — **{cp_sel['label']}** ({cp_sel['date']})."
    )

    model_sel = models[bk_current]
    df_sel = data[bk_current]
    _, original_cols = get_features(df_sel, bk_current)
    mapping = col_mapping(original_cols)

    st.subheader("Pes global de cada variable")
    importance = model_sel.feature_importance(importance_type="gain")
    feat_names = model_sel.feature_name()
    fi_df = pd.DataFrame({
        "feature": [shorten(mapping.get(f, f)) for f in feat_names],
        "importance": importance,
    }).sort_values("importance", ascending=False).head(15).sort_values("importance")

    fig = go.Figure(go.Bar(x=fi_df["importance"], y=fi_df["feature"], orientation="h",
                            marker_color="#1f78b4"))
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Pes en la decisió")
    st.plotly_chart(fig, use_container_width=True)

    # ── SHAP: càlcul comú per a tot el grup ────────────────────────────────
    sv, X_all, original_cols_all = compute_shap_matrix(bk_current, model_sel, df_sel)
    mapping_all = col_mapping(original_cols_all)
    probs_all = predict_all(model_sel, df_sel, bk_current)
    thr_all = threshold_for(bk_current)
    en_risc_all = probs_all <= thr_all

    mean_abs = pd.Series(abs(sv).mean(axis=0), index=X_all.columns)
    top_n = 12
    top_cols = mean_abs.sort_values(ascending=False).head(top_n).index.tolist()
    top_cols_asc = list(reversed(top_cols))  # el més important quedarà a dalt del gràfic

    st.divider()
    st.subheader("Visió de conjunt: com afecta cada variable a tot el grup")
    st.caption(
        "Cada punt és un estudiant. El color indica si el seu valor d'aquesta variable és alt (vermell) "
        "o baix (blau). Punts a la dreta empenyen cap a l'aprovat; a l'esquerra, cap al risc."
    )
    fig_bee = go.Figure()
    rng = pd.Series(range(len(X_all)))
    for i, col in enumerate(top_cols_asc):
        vals = X_all[col].values.astype(float)
        vmin, vmax = vals.min(), vals.max()
        norm = (vals - vmin) / (vmax - vmin) if vmax > vmin else pd.Series(vals).apply(lambda v: 0.5)
        jitter = pd.Series(range(len(vals))).apply(lambda j: ((j * 9301 + 49297) % 233280) / 233280.0)
        y_pos = i + (jitter - 0.5) * 0.7
        fig_bee.add_trace(go.Scatter(
            x=sv[:, X_all.columns.get_loc(col)], y=y_pos, mode="markers",
            marker=dict(
                color=norm, colorscale="RdBu_r", size=6, opacity=0.75,
                showscale=(i == len(top_cols_asc) - 1),
                colorbar=dict(title="Valor", x=1.02) if i == len(top_cols_asc) - 1 else None,
            ),
            showlegend=False, hoverinfo="skip",
        ))
    fig_bee.update_layout(
        height=70 * len(top_cols_asc) + 60,
        margin=dict(l=10, r=80, t=10, b=30),
        xaxis_title="Pes en la decisió (SHAP)",
        yaxis=dict(tickmode="array", tickvals=list(range(len(top_cols_asc))),
                    ticktext=[shorten(mapping_all[c]) for c in top_cols_asc]),
    )
    st.plotly_chart(fig_bee, use_container_width=True)

    st.divider()
    st.subheader("Què diferencia els estudiants en risc dels que no ho estan")
    top10 = mean_abs.sort_values(ascending=False).head(10).index.tolist()
    top10_asc = list(reversed(top10))
    mean_risc = [sv[en_risc_all, X_all.columns.get_loc(c)].mean() if en_risc_all.any() else 0 for c in top10_asc]
    mean_norisc = [sv[~en_risc_all, X_all.columns.get_loc(c)].mean() if (~en_risc_all).any() else 0 for c in top10_asc]
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(x=mean_risc, y=[shorten(mapping_all[c]) for c in top10_asc],
                              orientation="h", name="Estudiants en risc", marker_color="#e31a1c"))
    fig_cmp.add_trace(go.Bar(x=mean_norisc, y=[shorten(mapping_all[c]) for c in top10_asc],
                              orientation="h", name="Sense risc", marker_color="#33a02c"))
    fig_cmp.update_layout(height=480, barmode="group", margin=dict(l=10, r=10, t=20, b=10),
                           xaxis_title="Pes mitjà en la decisió (SHAP)",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.divider()
    st.subheader("Relació entre una variable i el risc")
    feat_choice_short = st.selectbox(
        "Variable a explorar", [shorten(mapping_all[c]) for c in top_cols],
    )
    feat_choice = top_cols[[shorten(mapping_all[c]) for c in top_cols].index(feat_choice_short)]
    idx_feat = X_all.columns.get_loc(feat_choice)
    fig_dep = go.Figure()
    fig_dep.add_trace(go.Scatter(
        x=X_all.loc[en_risc_all, feat_choice], y=sv[en_risc_all, idx_feat], mode="markers",
        name="En risc", marker=dict(color="#e31a1c", size=7, opacity=0.75),
    ))
    fig_dep.add_trace(go.Scatter(
        x=X_all.loc[~en_risc_all, feat_choice], y=sv[~en_risc_all, idx_feat], mode="markers",
        name="Sense risc", marker=dict(color="#33a02c", size=7, opacity=0.75),
    ))
    fig_dep.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10),
                           xaxis_title=f"Valor de la variable: {feat_choice_short}",
                           yaxis_title="Pes en la decisió (SHAP)",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_dep, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────
# PÀGINA 4 — Com funciona
# ──────────────────────────────────────────────────────────────────────────
else:
    st.title("Com funciona")
    st.markdown(
        """
Aquesta eina ajuda a identificar, amb antelació, els estudiants amb més risc de no
superar l'assignatura, a partir de la seva activitat a l'Aula Virtual i del seu
rendiment acadèmic parcial.

**Com interpretar les alertes**
- Cada estudiant rep una probabilitat estimada d'aprovar l'assignatura.
- Quan aquesta probabilitat cau per sota del llindar de risc, l'estudiant queda
  classificat **en risc** i apareix al Panell de seguiment.
- A la Fitxa de l'estudiant pots veure com ha evolucionat aquesta probabilitat al
  llarg del curs i quins factors hi influeixen més.

**Què fer amb una alerta**
Una alerta indica que convé parar atenció a aquell estudiant — per exemple,
contactar-lo o oferir-li suport — però no és un diagnòstic definitiu. La decisió
final correspon sempre al professorat.

**Privacitat**
Els identificadors dels estudiants estan anonimitzats: no permeten relacionar la
informació amb una persona concreta fora d'aquesta eina.
        """
    )
