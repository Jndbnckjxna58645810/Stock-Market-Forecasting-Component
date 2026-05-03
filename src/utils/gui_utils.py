import pandas as pd

def get_feature_table(metadata):
    macro_map = {m['name']: m['source'] for m in metadata.get('macro_features', [])}
    
    selected = metadata.get('selected_features', [])
    importances = metadata.get('feature_importances') or {}

    table_data = []
    
    for feat in selected:
        display_name = feat
        if feat in macro_map:
            display_name = f"{feat} ({macro_map[feat]})"

        imp_value = importances.get(feat, 0.0)
        
        if not importances:
            table_data.append({
                "Feature": display_name
            })
        else:
            table_data.append({
                "Feature": display_name,
                "Importance": imp_value
            })

    df_feat = pd.DataFrame(table_data)

    if importances:
        df_feat = df_feat.sort_values(by="Importance", ascending=False).reset_index(drop=True)
    
    return df_feat
