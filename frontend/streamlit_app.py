import os
import sys

# Make the repo root importable so `backend` resolves when Streamlit runs this file.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import graphviz

st.set_page_config(
    page_title="FinGraph AI",
    page_icon="📊",
    layout="wide"
)

# On Streamlit Cloud, secrets live in st.secrets. Copy them into the environment
# BEFORE importing backend modules (backend/config.py reads keys from the env).
try:
    for _k in ("PINECONE_API_KEY", "PINECONE_INDEX_NAME", "GROQ_API_KEY", "GROQ_MODEL"):
        if _k in st.secrets:
            os.environ.setdefault(_k, str(st.secrets[_k]))
except Exception:
    pass

from backend.service import (
    get_companies,
    run_compare,
    run_compare_kg,
    run_compare_fusion,
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #09090f 0%, #111827 45%, #020617 100%);
    color: white;
}

section[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #334155;
}

.main-title {
    font-size: 58px;
    font-weight: 900;
    line-height: 1.05;
    background: linear-gradient(90deg, #ffffff, #93c5fd, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    color: #94a3b8;
    font-size: 19px;
    margin-bottom: 35px;
}

.glass-card {
    background: rgba(15, 23, 42, 0.82);
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 22px;
    padding: 26px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.35);
}

.company-card {
    background: transparent;
    border: none;
    padding: 0px;
}

.metric-card {
    background: rgba(2, 6, 23, 0.8);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 18px;
    padding: 18px;
    text-align: center;
}

.small-label {
    color: #94a3b8;
    font-size: 14px;
}

.big-number {
    font-size: 34px;
    font-weight: 800;
    color: #bfdbfe;
}

.stButton > button {
    border-radius: 16px;
    height: 52px;
    font-weight: 700;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    border: none;
}

.stTextArea textarea {
    border-radius: 16px;
}
</style>
""", unsafe_allow_html=True)


def get_sentiment_df(nlp):
    sentiment = nlp.get("sentiment_summary", {})
    return pd.DataFrame({
        "Sentiment": ["Positive", "Negative", "Neutral"],
        "Chunks": [
            sentiment.get("positive_chunks", 0),
            sentiment.get("negative_chunks", 0),
            sentiment.get("neutral_chunks", 0)
        ]
    })


try:
    companies = get_companies()["companies"]
except Exception as e:
    st.error(f"Could not load data / connect to Pinecone. Check your API keys in Secrets. ({e})")
    st.stop()

if not companies:
    st.warning("No company files found in data folder.")
    st.stop()


with st.sidebar:
    st.markdown("## ⚙️ Filters")

    mode = st.radio(
        "Retrieval Mode",
        ["RAG", "KG", "Fusion"],
        index=0
    )

    if mode == "RAG":
        st.caption("Uses document chunks and vector search.")
    elif mode == "KG":
        st.caption("Uses company-entity relationships from extracted filing metadata.")
    else:
        st.caption("Combines document retrieval with Knowledge Graph entity relationships.")

    st.divider()

    st.markdown("## 🏢 Companies")

    company_a = st.selectbox("Company A", companies, index=0)

    company_b = st.selectbox(
        "Company B",
        companies,
        index=1 if len(companies) > 1 else 0
    )

    st.divider()

    st.markdown("## 📌 Sections")
    show_answer = st.checkbox("Answer", value=True)
    show_sentiment = st.checkbox("Sentiment", value=True)
    show_sources = st.checkbox("Sources", value=True)


st.markdown('<div class="main-title">FinGraph AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Compare companies using financial filings, retrieval intelligence, sentiment signals, and graph-ready evidence.</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)

st.subheader("Ask a comparison question")

st.markdown("### Demo Questions")

demo_questions = [
    "Compare these two companies based on business model, risks, revenue performance, and growth opportunities.",
    "Which company appears to have stronger growth opportunities?",
    "Compare the major risk factors of both companies.",
    "Which company has better revenue stability?",
    "Compare both companies based on Covid-19 impact."
]

selected_demo_question = st.selectbox(
    "Choose a demo question",
    demo_questions
)

st.markdown("### Ask Your Own Question")

custom_question = st.text_area(
    "Type your own question here",
    placeholder="Example: Which company looks financially stronger and why?",
    height=120
)

if custom_question.strip():
    question = custom_question
else:
    question = selected_demo_question

compare_clicked = st.button("🚀 Compare Companies", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

if compare_clicked:
    if company_a == company_b:
        st.warning("Please select two different companies.")
        st.stop()

    with st.spinner(f"Running {mode} analysis..."):
        try:
            if mode == "RAG":
                result = run_compare(company_a, company_b, question)
            elif mode == "KG":
                result = run_compare_kg(company_a, company_b, question)
            else:
                result = run_compare_fusion(company_a, company_b, question)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

    st.divider()

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="small-label">Retrieval Mode</div>
                <div class="big-number">{mode}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="small-label">{result['company_a'].title()} Evidence Score</div>
                <div class="big-number">{len(result.get('company_a_relationships', [])) if mode == "KG" else str(result.get('company_a_confidence', 0)) + "%"}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="small-label">{result['company_b'].title()} Evidence Score</div>
                <div class="big-number">{len(result.get('company_b_relationships', [])) if mode == "KG" else str(result.get('company_b_confidence', 0)) + "%"}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.info(
        "RAG uses vector search over filing chunks. KG uses extracted entity relationships. Fusion combines both."
)

    tab1, tab2, tab3, tab4 = st.tabs(["🧠 Answer", "📊 Sentiment", "📚 Sources", "🕸️ Graph"])

    with tab1:
        if show_answer:
            st.markdown("## 🧠 AI Comparison Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"### {result['company_a'].title()}")
                st.markdown('<div class="company-card">', unsafe_allow_html=True)
                st.write(result["company_a_answer"])
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown(f"### {result['company_b'].title()}")
                st.markdown('<div class="company-card">', unsafe_allow_html=True)
                st.write(result["company_b_answer"])
                st.markdown('</div>', unsafe_allow_html=True)

            st.divider()

            st.markdown("## 🏆 Final Verdict")
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.write(result.get("final_comparison", "No final comparison returned."))
            st.markdown('</div>', unsafe_allow_html=True)


    with tab2:
        if show_sentiment:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"### {result['company_a'].title()} Sentiment")
                df_a = get_sentiment_df(result.get("company_a_nlp", {}))
                st.bar_chart(df_a.set_index("Sentiment"))

            with col2:
                st.markdown(f"### {result['company_b'].title()} Sentiment")
                df_b = get_sentiment_df(result.get("company_b_nlp", {}))
                st.bar_chart(df_b.set_index("Sentiment"))

            st.caption("Sentiment is calculated from the retrieved SEC filing chunks used to answer the question.")

    with tab3:
        if mode == "KG":
            st.info("KG mode uses entity relationships instead of raw document source chunks. Use Fusion or RAG mode to view document sources.")
        else:
            if show_sources:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"### Sources for {result['company_a'].title()}")
                    for source in result["company_a_sources"]:
                        with st.expander(f"Page {source.get('page')} | Score {round(source.get('score', 0), 3)}"):
                            st.write(source.get("preview", ""))
                            st.write("Sentiment:", source.get("sentiment", "Not available"))
                            st.write("Entities:", source.get("entities", []))

                with col2:
                    st.markdown(f"### Sources for {result['company_b'].title()}")
                    for source in result["company_b_sources"]:
                        with st.expander(f"Page {source.get('page')} | Score {round(source.get('score', 0), 3)}"):
                            st.write(source.get("preview", ""))
                            st.write("Sentiment:", source.get("sentiment", "Not available"))
                            st.write("Entities:", source.get("entities", []))

    with tab4:
        st.markdown("## 🕸️ Knowledge Graph Visualization")
        st.caption("Graph nodes are generated from entities extracted from retrieved filing chunks.")

        if mode == "RAG":
            st.info("Graph visualization is available only in KG and Fusion modes.")
        else:
            graph_col1, graph_col2 = st.columns(2)

        with graph_col1:
            st.markdown(f"### {result['company_a'].title()}")

            dot = graphviz.Digraph()
            dot.attr(rankdir="LR")

            company_name = result["company_a"].title()

            dot.node(
                company_name,
                shape="box",
                style="filled",
                fillcolor="lightblue"
            )

            entities = result.get("company_a_entities", [])[:10]

            for item in entities:
                entity = item.get("entity", "")
                count = item.get("count", 1)

                if entity:
                    dot.node(entity)
                    dot.edge(
                        company_name,
                        entity,
                        label=f"MENTIONS ({count})"
                    )

            st.graphviz_chart(dot, use_container_width=True)

        with graph_col2:
            st.markdown(f"### {result['company_b'].title()}")

            dot = graphviz.Digraph()
            dot.attr(rankdir="LR")

            company_name = result["company_b"].title()

            dot.node(
                company_name,
                shape="box",
                style="filled",
                fillcolor="lightgreen"
            )

            entities = result.get("company_b_entities", [])[:10]

            for item in entities:
                entity = item.get("entity", "")
                count = item.get("count", 1)

                if entity:
                    dot.node(entity)
                    dot.edge(
                        company_name,
                        entity,
                        label=f"MENTIONS ({count})"
                    )

            st.graphviz_chart(dot, use_container_width=True)

        st.divider()

        st.markdown("## 📋 Graph Relationships")

        rel_col1, rel_col2 = st.columns(2)

        with rel_col1:
            relationships_a = result.get("company_a_relationships", [])

            if relationships_a:
                st.dataframe(
                    pd.DataFrame(relationships_a),
                    use_container_width=True
                )
            else:
                st.info("No relationships found.")

        with rel_col2:
            relationships_b = result.get("company_b_relationships", [])

            if relationships_b:
                st.dataframe(
                    pd.DataFrame(relationships_b),
                    use_container_width=True
                )
            else:
                st.info("No relationships found.")