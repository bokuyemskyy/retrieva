from __future__ import annotations

from contextlib import contextmanager
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class GPUModel(Protocol):
    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass


class GPUManager:
    def __init__(self) -> None:
        self._current: Optional[GPUModel] = None

    @property
    def active_model(self) -> Optional[GPUModel]:
        return self._current

    def activate(self, model: GPUModel) -> None:
        if self._current is model:
            return

        if self._current is not None:
            self._current.unload()
            self._current = None

        model.load()
        self._current = model

    def deactivate(self) -> None:
        if self._current is not None:
            self._current.unload()
            self._current = None

    @contextmanager
    def using(self, model: GPUModel):
        self.activate(model)
        try:
            yield
        finally:
            if self._current is model:
                self.deactivate()
