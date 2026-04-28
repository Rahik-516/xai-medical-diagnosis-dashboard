"""Heart disease prediction page."""

from __future__ import annotations

import streamlit as st

from config.paths import get_task_manifest, resolve_path
from src.components.disclaimer import render_disclaimer
from src.components.figure_display import display_image_group
from src.components.header import render_page_header
from src.components.metric_cards import render_prediction_metrics
from src.explainability.explanation_renderer import render_tabular_explanation_graph
from src.explainability.fallback_loader import get_tabular_fallback_image
from src.explainability.explanation_rules import generate_tabular_explanation
from src.explainability.feature_name_mapping import humanize_feature
from src.models.heart_model_loader import load_heart_artifacts
from src.pipelines.heart_pipeline import predict_heart
from src.explainability.shap_runtime import generate_local_shap_figure
from src.utils.logger import get_logger
from src.utils.session_state import save_prediction

LOGGER = get_logger(__name__)


def _render_heart_form() -> dict[str, float]:
    """Render heart form controls and return payload."""
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("age", min_value=20.0,
                              max_value=100.0, value=55.0, step=1.0)
        sex = st.selectbox("sex", options=[0, 1], index=1)
        cp = st.selectbox("cp", options=[0, 1, 2, 3], index=0)
        trestbps = st.number_input(
            "trestbps", min_value=70.0, max_value=220.0, value=130.0, step=1.0)
        chol = st.number_input("chol", min_value=100.0,
                               max_value=700.0, value=240.0, step=1.0)
    with col2:
        fbs = st.selectbox("fbs", options=[0, 1], index=0)
        restecg = st.selectbox("restecg", options=[0, 1, 2], index=1)
        thalach = st.number_input(
            "thalach", min_value=60.0, max_value=250.0, value=150.0, step=1.0)
        exang = st.selectbox("exang", options=[0, 1], index=0)
        oldpeak = st.number_input(
            "oldpeak", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    with col3:
        slope = st.selectbox("slope", options=[0, 1, 2], index=1)
        ca = st.selectbox("ca", options=[0, 1, 2, 3, 4], index=0)
        thal = st.selectbox("thal", options=[0, 1, 2, 3], index=2)

    return {
        "age": age,
        "sex": float(sex),
        "cp": float(cp),
        "trestbps": trestbps,
        "chol": chol,
        "fbs": float(fbs),
        "restecg": float(restecg),
        "thalach": thalach,
        "exang": float(exang),
        "oldpeak": oldpeak,
        "slope": float(slope),
        "ca": float(ca),
        "thal": float(thal),
    }


def render_heart_page() -> None:
    """Render heart disease prediction workflow page."""
    render_page_header("Heart Disease Prediction",
                       "Pipeline-based inference with engineered features")
    render_disclaimer()

    st.subheader("Patient Inputs")
    payload = _render_heart_form()

    if st.button("Predict Heart Disease Risk", type="primary"):
        try:
            result = predict_heart(payload)
            render_prediction_metrics(
                result.label, result.probability, result.confidence_band)

            st.subheader("Local Explanation")
            positive_text = ", ".join(humanize_feature(item)
                                      for item in result.positive_contributors)
            negative_text = ", ".join(humanize_feature(item)
                                      for item in result.negative_contributors)
            st.write(
                f"Key risk-increasing factors: {positive_text if positive_text else 'Not identified'}")
            st.write(
                f"Potential risk-reducing factors: {negative_text if negative_text else 'Not identified'}")

            summary = generate_tabular_explanation(
                task_name="Heart Disease Prediction",
                predicted_label=result.label,
                predicted_probability=result.probability,
                confidence_band=result.confidence_band,
                positive_contributors=result.positive_contributors,
                negative_contributors=result.negative_contributors,
            )
            st.subheader("Rule-Based Clinical Summary")
            st.info(summary)

            try:
                artifacts = load_heart_artifacts()
                shap_result = generate_local_shap_figure(
                    task_key="heart",
                    model=artifacts["pipeline"],
                    model_input=result.model_input,
                )
                fallback_image = get_tabular_fallback_image("heart")
                render_tabular_explanation_graph(
                    task_name="Heart Disease Prediction",
                    shap_result=shap_result,
                    fallback_path=fallback_image,
                )
            except Exception:  # noqa: BLE001
                LOGGER.exception("Heart explanation rendering failed.")
                st.warning(
                    "Prediction completed successfully, but the explanation graph is unavailable for this input."
                )

            save_prediction(
                task="heart",
                label=result.label,
                probability=result.probability,
                confidence_band=result.confidence_band,
                summary=summary,
            )
            LOGGER.info("Rendered heart prediction result.")
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Heart prediction failed.")
            st.error(f"Heart prediction failed: {exc}")

    task_manifest = get_task_manifest("heart")
    st.subheader("Static Explainability Figures")
    required_figures = [
        resolve_path("static_outputs/heart/shap_bar_Heart.png"),
        resolve_path("static_outputs/heart/shap_waterfall_Heart_s1.png"),
        resolve_path("static_outputs/heart/lime_Heart_HighRisk_Positive.png"),
    ]
    display_image_group(
        required_figures,
        columns=2,
        image_width=560,
        use_container_width=True,
    )

    extra_xai = [resolve_path(path)
                 for path in task_manifest["static_figures"]["xai"]]
    with st.expander("Show additional heart XAI figures"):
        display_image_group(
            extra_xai,
            columns=2,
            image_width=520,
            use_container_width=True,
        )

    render_disclaimer()
