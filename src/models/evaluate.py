import pandas as pd

def get_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        return (
            pd.Series(model.feature_importances_, index=feature_names)
            .sort_values(ascending=False))
    else: return None