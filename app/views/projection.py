import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.preprocessing import normalize
import umap


@st.dialog("Chunk Details")
def _chunk_inspector_dialog(chunk_data: dict):
    """Displays chunk details in a clean popup modal."""
    st.markdown(f"**Source File:** `{chunk_data.get('filename', 'Unknown')}`")
    st.markdown(f"**Modality:** `{chunk_data.get('modality', 'Unknown')}`")
    st.markdown(f"**Chunk ID:** `{chunk_data['chunk_id']}`")

    metadata = chunk_data.get("metadata", {})
    if metadata:
        with st.expander("View Metadata"):
            st.json(metadata)

    st.markdown("**Content:**")
    st.text_area(
        "Content",
        value=chunk_data["content"],
        height=300,
        # Removed disabled=True to get rid of the gray background
        label_visibility="collapsed",
    )


def render_projection():
    st.title("Chunk Projection")
    st.divider()

    ws = st.session_state.active_ws
    if not ws:
        st.info("Select or create a workspace in the sidebar.")
        return

    with st.spinner("Fetching data..."):
        try:
            chunks = st.session_state.rag.get_chunks_with_embeddings(
                workspace_name=ws, limit=10_000
            )
            # Fetch documents to map document_id to filename
            docs = st.session_state.rag.get_documents(workspace_name=ws)
            doc_map = {str(d.document_id): d.filename for d in docs}
        except Exception as e:
            st.error(f"Error loading data: {e}")
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

    # Build the dataframe with the new requested fields
    df = pd.DataFrame(
        {
            "x": projection[:, 0],
            "y": projection[:, 1],
            "chunk_id": [str(c.chunk_id) for c in valid_chunks],
            "document_id": [str(c.document_id) for c in valid_chunks],
            "filename": [
                doc_map.get(str(c.document_id), "Unknown") for c in valid_chunks
            ],
            "modality": [
                c.modality.value if hasattr(c.modality, "value") else str(c.modality)
                for c in valid_chunks
            ],
            "metadata": [c.metadata for c in valid_chunks],
            "content": [c.content for c in valid_chunks],
            "snippet": [c.content[:80] + "..." for c in valid_chunks],
        }
    )

    fig = px.scatter(
        df,
        x="x",
        y="y",
        hover_name="snippet",
        custom_data=["chunk_id"],
        opacity=0.7,
    )

    # Prevent dimming on selection and format the hover tooltip
    fig.update_traces(
        marker=dict(size=8, line=dict(width=1, color="DarkSlateGrey")),
        selected=dict(marker=dict(opacity=0.7)),
        unselected=dict(marker=dict(opacity=0.7)),
        hovertemplate="<b>%{hovertext}</b><extra></extra>",
    )

    # Clean up layout, titles, and hover styling
    fig.update_layout(
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        hoverlabel=dict(
            font_size=12,
            font_family="sans-serif",
            bordercolor="rgba(0,0,0,0.1)",
        ),
        hovermode="closest",
        clickmode="event+select",
        dragmode="pan",
    )

    event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
    )

    # Trigger the modal dialog if a point is selected
    if event and event.selection and event.selection.points:
        idx = event.selection.points[0]["point_index"]
        selected_data = df.iloc[idx].to_dict()
        _chunk_inspector_dialog(selected_data)
