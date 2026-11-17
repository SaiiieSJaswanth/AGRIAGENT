import pandas as pd


def load_state_crop_catalog(path):
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=["state", "crop", "crop_category", "source_basis"])
    return df


def crops_for_state(catalog, state):
    if catalog.empty:
        return []
    subset = catalog[catalog["state"].astype(str).str.casefold() == str(state).casefold()]
    return subset[["crop", "crop_category"]].drop_duplicates().to_dict("records")
