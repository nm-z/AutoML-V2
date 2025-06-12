from __future__ import annotations

from sklearn.tree import DecisionTreeRegressor as _DTR


class DecisionTree(_DTR):
    """Expose sklearn's DecisionTreeRegressor under project namespace.""" 