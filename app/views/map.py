import streamlit as st


def render_map():
    st.info("Not implemented.")
    return

    # ws = st.session_state.active_ws

    # col_title, _ = st.columns([4, 1])
    # with col_title:
    #     st.markdown(
    #         f'<p class="mono" style="font-size:0.75rem;color:var(--text-muted);letter-spacing:0.08em;margin-bottom:0;">EMBEDDING MAP</p>'
    #         f'<h2 style="margin-top:0;margin-bottom:1.5rem;">{ws or "—"}</h2>',
    #         unsafe_allow_html=True,
    #     )

    # if not ws:
    #     st.info("Select a workspace first.")
    #     return

    # method_col, run_col, _ = st.columns([2, 1, 5])
    # with method_col:
    #     method = st.selectbox(
    #         "Projection", ["UMAP", "t-SNE", "PCA"], label_visibility="collapsed"
    #     )
    # with run_col:
    #     run = st.button("Compute", type="primary")

    # st.divider()

    # if run:
    #     with st.spinner("Fetching embeddings and projecting…"):
    #         try:
    #             import numpy as np

    #             embeddings_data = st.session_state.rag.get_all_embeddings()
    #             vectors = np.array([e["vector"] for e in embeddings_data])
    #             labels = [
    #                 e.get("source", f"chunk_{i}") for i, e in enumerate(embeddings_data)
    #             ]

    #             if len(vectors) < 2:
    #                 st.warning("Not enough embeddings to project (need at least 2).")
    #                 return

    #             if method == "UMAP":
    #                 from umap import UMAP

    #                 reducer = UMAP(n_components=2, random_state=42)
    #                 coords = reducer.fit_transform(vectors)
    #             elif method == "t-SNE":
    #                 from sklearn.manifold import TSNE

    #                 perplexity = min(30, len(vectors) - 1)
    #                 reducer = TSNE(
    #                     n_components=2, random_state=42, perplexity=perplexity
    #                 )
    #                 coords = reducer.fit_transform(vectors)
    #             else:  # PCA
    #                 from sklearn.decomposition import PCA

    #                 reducer = PCA(n_components=2)
    #                 coords = reducer.fit_transform(vectors)

    #             import plotly.express as px
    #             import pandas as pd

    #             df = pd.DataFrame(
    #                 {
    #                     "x": coords[:, 0],
    #                     "y": coords[:, 1],
    #                     "source": labels,
    #                 }
    #             )

    #             fig = px.scatter(
    #                 df,
    #                 x="x",
    #                 y="y",
    #                 color="source",
    #                 hover_data={"source": True, "x": False, "y": False},
    #                 template="plotly_dark",
    #             )
    #             fig.update_layout(
    #                 paper_bgcolor="#0f0f11",
    #                 plot_bgcolor="#16161a",
    #                 font_family="IBM Plex Mono",
    #                 legend_title_text="Source",
    #                 margin=dict(l=0, r=0, t=20, b=0),
    #                 height=520,
    #             )
    #             fig.update_traces(marker=dict(size=5, opacity=0.8))
    #             st.plotly_chart(fig, use_container_width=True)

    #         except Exception as e:
    #             st.error(f"Projection failed: {e}")
    # else:
    #     st.markdown(
    #         '<p style="color:var(--text-muted);font-size:0.88rem;">Choose a projection method and click Compute to visualize your embeddings.</p>',
    #         unsafe_allow_html=True,
    #     )
