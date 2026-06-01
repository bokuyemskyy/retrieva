import tempfile
import os


def save_upload_to_tmp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def cfg_display_name(provider: str, model_name: str) -> str:
    return f"{model_name} ({provider})"
