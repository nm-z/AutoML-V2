from __future__ import annotations

from sklearn.svm import SVR as _SVR


class SVR(_SVR):
    """Expose sklearn's SVR under project namespace.""" 