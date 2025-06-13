from __future__ import annotations

"""Base abstractions for all *concrete* AutoML components.

Every model or preprocessing block **must** inherit from either
`BaseEstimatorBlock` (for estimators with `fit`/`predict`) or
`BaseTransformerBlock` (for stateless/stateless* preprocessors that expose
`fit`/`transform`).  These mixins exist purely to enforce a *uniform* public
API across the heterogeneous wrappers scattered throughout the harness.

Every AutoML engine wrapper **must** inherit from `BaseEngine` to ensure a
consistent public API for the orchestrator to interact with.

The classes intentionally avoid any heavy lifting or additional dependencies –
they simply hard-code the minimal method signatures expected by the orches-
trator so that type-checkers (mypy/Pyright) and static linters can reason
about the call-sites.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseComponent(ABC):
    """Root of the component hierarchy – do *not* subclass directly."""

    # Each concrete block *must* override this with a concise description of
    # its nature and hyper-parameters.  The orchestrator relies on this during
    # provenance logging and JSON serialisation of the champion pipeline.
    signature: Dict[str, Any] = {}

    @classmethod
    def get_signature(cls) -> Dict[str, Any]:
        """Return the static *signature* of the block. This is used by
orchestrator logging utilities to embed provenance information.
"""
        return dict(cls.signature)


class BaseEstimatorBlock(BaseComponent):
    """Mixin for estimator-style components (expose `fit`/`predict`)."""

    @abstractmethod
    def fit(self, X, y=None):
        """Fit the estimator to *X* and *y*."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        """Predict targets for *X*."""
        raise NotImplementedError


class BaseTransformerBlock(BaseComponent):
    """Mixin for transformer-style components (expose `fit`/`transform`)."""

    @abstractmethod
    def fit(self, X, y=None):
        raise NotImplementedError

    @abstractmethod
    def transform(self, X):
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        """Utility to fit *and* transform in one pass."""
        self.fit(X, y)
        return self.transform(X)


class BaseEngine(ABC):
    """Base class for all AutoML engine wrappers."""

    @abstractmethod
    def fit(self, X, y):
        """Fit the AutoML engine."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        """Predict targets using the fitted AutoML engine."""
        raise NotImplementedError

    @abstractmethod
    def export(self, file_path: str):
        """Export the champion pipeline to a file."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the engine."""
        raise NotImplementedError

    @property
    @abstractmethod
    def best_pipeline_info(self) -> dict:
        """Return information about the best pipeline found by the engine."""
        raise NotImplementedError

    @property
    @abstractmethod
    def run_info(self) -> dict:
        """Return information about the engine's run, including artifacts and logs."""
        raise NotImplementedError 
