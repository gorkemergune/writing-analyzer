import plotly.graph_objects as go
import streamlit as st


def risk_color(score: float) -> str:
    if score <= 30:
        return "#27ae60"
    if score <= 55:
        return "#f39c12"
    if score <= 75:
        return "#e67e22"
    return "#e74c3c"


def risk_label(risk_level: str, t: dict) -> str:
    mapping = {
        "low": t["risk_low"],
        "moderate": t["risk_moderate"],
        "high": t["risk_high"],
        "very_high": t["risk_very_high"],
    }
    return mapping.get(risk_level, risk_level)


def render_risk_summary(academic_risk: dict, t: dict) -> None:
    score = academic_risk["overall_score"]
    level = academic_risk["risk_level"]
    confidence = academic_risk["confidence"]
    color = risk_color(score)
    label = risk_label(level, t)

    col_score, col_conf = st.columns([1, 2])
    with col_score:
        st.markdown(
            f"""
            <div style="
                background:{color}12;
                border:2px solid {color};
                border-radius:14px;
                padding:28px 20px;
                text-align:center;
            ">
                <div style="font-size:3.8rem;font-weight:900;color:{color};line-height:1">
                    {score:.1f}
                </div>
                <div style="font-size:0.8rem;color:#888;margin-top:4px">
                    {t['risk_score_title']}
                </div>
                <div style="margin-top:14px">
                    <span style="
                        background:{color};
                        color:white;
                        padding:5px 14px;
                        border-radius:20px;
                        font-size:0.8rem;
                        font-weight:600;
                    ">{label}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_conf:
        st.metric(t["confidence_title"], f"{confidence:.0%}")
        st.caption(f"{t['risk_level_title']}: **{label}**")


def render_stats_row(report: dict, t: dict) -> None:
    ws = report["word_stats"]
    ss = report["sentence_stats"]
    pt = report["processing_time_ms"]

    cols = st.columns(6)
    stats = [
        (t["word_count"], str(ws["total_words"])),
        (t["sentence_count"], str(ss["total_sentences"])),
        (t["unique_words"], str(ws["unique_words"])),
        (t["lexical_diversity"], f"{ws['lexical_diversity']:.2f}"),
        (t["avg_sentence_len"], f"{ss['avg_sentence_length']:.1f}"),
        (t["processing_time"], f"{pt:.0f} {t['ms']}"),
    ]
    for col, (label, value) in zip(cols, stats, strict=False):
        with col:
            st.metric(label, value)


def render_radar_chart(component_scores: dict, t: dict) -> go.Figure:
    categories = [
        t["component_repetition"],
        t["component_transitions"],
        t["component_burstiness"],
        t["component_readability"],
        t["component_cliches"],
        t["component_lexical"],
    ]
    values = [
        component_scores["repetition"],
        component_scores["transition_overuse"],
        component_scores["low_burstiness"],
        component_scores["readability"],
        component_scores["cliche_density"],
        component_scores["lexical_poverty"],
    ]

    fig = go.Figure(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            line_color="#4A90D9",
            fillcolor="rgba(74, 144, 217, 0.18)",
            name="",
        )
    )
    fig.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 100]}},
        showlegend=False,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        height=360,
    )
    return fig


def render_component_cards(report: dict, t: dict) -> None:
    st.subheader(t["detail_subtitle"])
    cs = report["academic_risk"]["component_scores"]
    rep = report["repetition"]
    tr = report["transitions"]
    bu = report["burstiness"]
    rd = report["readability"]
    cl = report["cliches"]
    ws = report["word_stats"]

    phrase_count = len(rep["repeated_phrases"])
    phrase_detail = f"{phrase_count} {t['repeated_phrases']}" if phrase_count else t["none_detected"]

    tr_count = tr["transition_count"]
    tr_detail = f"{tr_count} {t['transitions_found']}" if tr_count else t["none_detected"]

    cliche_count = cl["cliche_count"]
    cliche_detail = (
        f"{cl['cliche_density']:.2f} {t['cliche_density']}" if cliche_count else t["none_detected"]
    )

    components: list[tuple[str, float, str]] = [
        (t["component_repetition"], cs["repetition"], phrase_detail),
        (t["component_transitions"], cs["transition_overuse"], tr_detail),
        (t["component_burstiness"], cs["low_burstiness"], bu["classification"]),
        (t["component_readability"], cs["readability"], f"{rd['readability_score']:.1f} — {rd['classification']}"),
        (t["component_cliches"], cs["cliche_density"], cliche_detail),
        (t["component_lexical"], cs["lexical_poverty"], f"{ws['lexical_diversity']:.2f}"),
    ]

    cols = st.columns(3)
    for i, (name, score, detail) in enumerate(components):
        color = risk_color(score)
        with cols[i % 3]:
            st.markdown(
                f"""
                <div style="
                    background:#fafafa;
                    border:1px solid #eee;
                    border-radius:10px;
                    padding:16px;
                    margin-bottom:12px;
                    border-top:3px solid {color};
                ">
                    <div style="font-weight:600;font-size:0.9rem;color:#444;margin-bottom:6px">
                        {name}
                    </div>
                    <div>
                        <span style="font-size:1.8rem;font-weight:800;color:{color}">{score:.0f}</span>
                        <span style="font-size:0.85rem;color:#aaa">/100</span>
                    </div>
                    <div style="font-size:0.78rem;color:#888;margin-top:4px">{detail}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_explanations(explanations: list, t: dict) -> None:
    st.markdown(f"**{t['explanations_title']}**")
    if not explanations:
        st.success(t["no_issues"])
        return
    for explanation in explanations:
        st.warning(explanation)


def render_suggestions(suggestions: list, t: dict) -> None:
    if not suggestions:
        return
    st.markdown(f"**{t['suggestions_title']}**")
    for suggestion in suggestions:
        st.info(suggestion)
