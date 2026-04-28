"""Diabetes prediction page."""

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
from src.explainability.shap_runtime import generate_local_shap_figure
from src.models.diabetes_model_loader import load_diabetes_artifacts
from src.pipelines.diabetes_pipeline import predict_diabetes
from src.utils.logger import get_logger
from src.utils.session_state import save_prediction

LOGGER = get_logger(__name__)


def _render_diabetes_form() -> dict[str, float]:
    """Render diabetes input controls and return payload."""
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input(
            "Pregnancies", min_value=0.0, max_value=20.0, value=1.0, step=1.0)
        glucose = st.number_input(
            "Glucose", min_value=0.0, max_value=300.0, value=120.0, step=1.0)
        blood_pressure = st.number_input(
            "BloodPressure", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
        skin_thickness = st.number_input(
            "SkinThickness", min_value=0.0, max_value=120.0, value=20.0, step=1.0)
    with col2:
        insulin = st.number_input(
            "Insulin", min_value=0.0, max_value=900.0, value=80.0, step=1.0)
        bmi = st.number_input("BMI", min_value=0.0,
                              max_value=80.0, value=28.0, step=0.1)
        dpf = st.number_input(
            "DiabetesPedigreeFunction",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.01,
            format="%.3f",
        )
        age = st.number_input("Age", min_value=20.0,
                              max_value=100.0, value=45.0, step=1.0)

    return {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }


def render_diabetes_page() -> None:
    """Render diabetes prediction workflow page."""
    render_page_header("Diabetes Prediction",
                       "Tabular model inference with rule-based explanation")
    render_disclaimer()

    st.subheader("Patient Inputs")
    payload = _render_diabetes_form()

    if st.button("Predict Diabetes Risk", type="primary"):
        try:
            result = predict_diabetes(payload)
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
                task_name="Diabetes Risk Prediction",
                predicted_label=result.label,
                predicted_probability=result.probability,
                confidence_band=result.confidence_band,
                positive_contributors=result.positive_contributors,
                negative_contributors=result.negative_contributors,
            )
            st.subheader("Rule-Based Clinical Summary")
            st.info(summary)

            try:
                artifacts = load_diabetes_artifacts()
                shap_result = generate_local_shap_figure(
                    task_key="diabetes",
                    model=artifacts["model"],
                    model_input=result.scaled_features,
                )
                fallback_image = get_tabular_fallback_image("diabetes")
                render_tabular_explanation_graph(
                    task_name="Diabetes Risk Prediction",
                    shap_result=shap_result,
                    fallback_path=fallback_image,
                )
            except Exception:  # noqa: BLE001
                LOGGER.exception("Diabetes explanation rendering failed.")
                st.warning(
                    "Prediction completed successfully, but the explanation graph is unavailable for this input."
                )

            save_prediction(
                task="diabetes",
                label=result.label,
                probability=result.probability,
                confidence_band=result.confidence_band,
                summary=summary,
            )
            LOGGER.info("Rendered diabetes prediction result.")
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Diabetes prediction failed.")
            st.error(f"Diabetes prediction failed: {exc}")

    task_manifest = get_task_manifest("diabetes")
    st.subheader("Static Explainability Figures")
    required_figures = [
        resolve_path("static_outputs/diabetes/shap_bar_Diabetes.png"),
        resolve_path("static_outputs/diabetes/shap_waterfall_Diabetes_s1.png"),
        resolve_path(
            "static_outputs/diabetes/lime_Diabetes_HighRisk_Positive.png"),
    ]
    display_image_group(
        required_figures,
        columns=2,
        image_width=560,
        use_container_width=True,
    )

    extra_xai = [resolve_path(path)
                 for path in task_manifest["static_figures"]["xai"]]
    with st.expander("Show additional diabetes XAI figures"):
        display_image_group(
            extra_xai,
            columns=2,
            image_width=520,
            use_container_width=True,
        )

    render_disclaimer()
