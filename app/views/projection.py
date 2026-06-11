import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.preprocessing import normalize
import umap


def render_projection():
    st.title("Chunk Projection (2D)")
    st.divider()

    ws = st.session_state.active_ws
    if not ws:
        st.info("Select or create a workspace in the sidebar.")
        return

    with st.spinner("Fetching chunks..."):
        try:
            chunks = st.session_state.rag.get_chunks_with_embeddings(
                workspace_name=ws, limit=10_000
            )
        except Exception as e:
            st.error(f"Error loading chunks: {e}")
            return

    if not chunks:
        st.info("No chunks available in this workspace to visualize.")
        return

    valid_chunks = [c.chunk for c in chunks]
    valid_embeddings = [c.embedding for c in chunks]

    embeddings = normalize(np.array(valid_embeddings))

    with st.spinner("Reducing dimensions (UMAP)..."):
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=20,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )

        projection = reducer.fit_transform(embeddings)

    df = pd.DataFrame(
        {
            "x": projection[:, 0],
            "y": projection[:, 1],
            "chunk_id": [str(c.chunk_id) for c in valid_chunks],
            "document_id": [str(c.document_id) for c in valid_chunks],
            "content": [c.content for c in valid_chunks],
            "snippet": [c.content[:80] + "..." for c in valid_chunks],
        }
    )

    if "selected_chunk" not in st.session_state:
        st.session_state.selected_chunk = None

    fig = px.scatter(
        df,
        x="x",
        y="y",
        hover_name="snippet",
        custom_data=["chunk_id"],
        title="Semantic Clusters of Document Chunks",
        opacity=0.7,
    )

    fig.update_traces(
        marker=dict(size=8, line=dict(width=1, color="DarkSlateGrey")),
        hovertemplate="<b>%{hovertext}</b><extra></extra>",
    )

    fig.update_layout(
        xaxis_title="Component 1",
        yaxis_title="Component 2",
    )

    event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
    )

    selected = None

    if event and event.selection and event.selection.points:
        idx = event.selection.points[0]["point_index"]
        selected = df.iloc[idx].to_dict()
        st.session_state.selected_chunk = selected

    selected = st.session_state.selected_chunk

    st.divider()
    st.subheader("Selected Chunk Inspector")

    if selected:
        st.markdown(f"**Document:** `{selected['document_id']}`")
        st.markdown(f"**Chunk ID:** `{selected['chunk_id']}`")
        st.text_area(
            "Full content",
            value=selected["content"],
            height=300,
        )
    else:
        st.info("Click a point in the plot to inspect its chunk.")
