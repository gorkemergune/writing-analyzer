import streamlit as st
from api_client import analyze_text, health_check
from components import (
    render_component_cards,
    render_explanations,
    render_radar_chart,
    render_risk_summary,
    render_stats_row,
    render_suggestions,
)
from translations import TRANSLATIONS

_DOC_TYPE_VALUES = ["essay", "academic", "email", "report", "assignment"]
_DOC_TYPE_KEYS = ["doc_essay", "doc_academic", "doc_email", "doc_report", "doc_assignment"]

_LANG_VALUES: list[str | None] = [None, "tr", "en"]
_LANG_KEYS = ["lang_auto", "lang_tr", "lang_en"]


def _t(key: str) -> str:
    return TRANSLATIONS[st.session_state.get("ui_lang", "tr")][key]


def _init_session() -> None:
    defaults: dict = {
        "ui_lang": "tr",
        "result": None,
        "doc_type_idx": 0,
        "text_lang_idx": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## Academic Writing Auditor")
        st.caption("v0.1.0 — MVP")
        st.divider()

        lang_choice = st.radio(
            _t("lang_label"),
            options=["🇹🇷 Türkçe", "🇬🇧 English"],
            index=0 if st.session_state.ui_lang == "tr" else 1,
            horizontal=True,
        )
        st.session_state.ui_lang = "tr" if "Türkçe" in lang_choice else "en"

        t = TRANSLATIONS[st.session_state.ui_lang]

        st.divider()

        st.selectbox(
            t["doc_type_label"],
            options=list(range(5)),
            format_func=lambda i: t[_DOC_TYPE_KEYS[i]],
            key="doc_type_idx",
        )

        st.selectbox(
            t["lang_detect_label"],
            options=list(range(3)),
            format_func=lambda i: t[_LANG_KEYS[i]],
            key="text_lang_idx",
        )

        st.divider()

        st.caption(t["backend_status"])
        try:
            health_check()
            st.success(t["connected"])
        except Exception:
            st.error(t["not_connected"])


def main() -> None:
    st.set_page_config(
        page_title="Academic Writing Auditor",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _init_session()
    _render_sidebar()

    t = TRANSLATIONS[st.session_state.ui_lang]

    st.title(t["title"])
    st.caption(t["subtitle"])
    st.divider()

    text: str = st.text_area(
        label=t["text_label"],
        placeholder=t["text_placeholder"],
        height=300,
        key="user_text",
    )

    col_analyze, col_clear, col_pad = st.columns([1, 1, 8])
    with col_analyze:
        analyze_clicked = st.button(t["analyze_btn"], type="primary", use_container_width=True)
    with col_clear:
        if st.button(t["clear_btn"], use_container_width=True):
            st.session_state.result = None
            st.session_state.user_text = ""
            st.rerun()

    if analyze_clicked:
        if len(text.strip()) < 50:
            st.error(t["error_min_chars"])
        else:
            doc_api = _DOC_TYPE_VALUES[st.session_state.doc_type_idx]
            lang_api = _LANG_VALUES[st.session_state.text_lang_idx]
            with st.spinner(t["loading"]):
                try:
                    st.session_state.result = analyze_text(text, doc_api, lang_api)
                except Exception:
                    st.error(t["error_api"])

    if st.session_state.result:
        result = st.session_state.result
        t = TRANSLATIONS[st.session_state.ui_lang]

        st.divider()
        st.subheader(t["results_title"])

        render_risk_summary(result["academic_risk"], t)
        st.write("")
        render_stats_row(result, t)

        st.divider()

        col_radar, col_signals = st.columns([1, 1])
        with col_radar:
            st.subheader(t["profile_title"])
            fig = render_radar_chart(result["academic_risk"]["component_scores"], t)
            st.plotly_chart(fig, use_container_width=True)
        with col_signals:
            st.write("")
            st.write("")
            render_explanations(result["academic_risk"]["explanations"], t)
            render_suggestions(result["suggestions"], t)

        st.divider()
        render_component_cards(result, t)


if __name__ == "__main__":
    main()
