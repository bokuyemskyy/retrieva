import streamlit as st
from ui.modals import create_workspace_dialog


NAV_ITEMS = [
    ("chat", "Chat"),
    ("files", "Files"),
    ("map", "Map"),
]


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            '<p class="mono" style="font-size:1.1rem;font-weight:600;letter-spacing:0.06em;margin-bottom:1.5rem;">RETRIEVA</p>',
            unsafe_allow_html=True,
        )

        st.markdown('<p class="section-label">Workspace</p>', unsafe_allow_html=True)

        ws_names = st.session_state.workspaces
        if ws_names:
            idx = (
                ws_names.index(st.session_state.active_ws)
                if st.session_state.active_ws in ws_names
                else 0
            )
            selected = st.selectbox(
                "workspace",
                options=ws_names,
                index=idx,
                label_visibility="collapsed",
            )
            if selected != st.session_state.active_ws:
                st.session_state.active_ws = selected
                try:
                    st.session_state.rag.select_workspace(selected)
                except Exception:
                    pass
                st.rerun()
        else:
            st.caption("No workspaces yet.")

        if st.button("New workspace", use_container_width=True):
            create_workspace_dialog()

        st.divider()

        st.markdown('<p class="section-label">Navigate</p>', unsafe_allow_html=True)

        current = st.session_state.active_view
        for key, label in NAV_ITEMS:
            active_style = (
                "color: var(--accent); font-weight: 600;" if key == current else ""
            )
            if st.button(
                label,
                key=f"nav_{key}",
                use_container_width=True,
            ):
                st.session_state.active_view = key
                st.rerun()

    return st.session_state.active_view
