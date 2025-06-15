import pandas as pd


def remove_duplicates(cleaned_data):
    data_train = cleaned_data[cleaned_data["split"] == "train"].copy()
    val_captions = set(cleaned_data[cleaned_data["split"] == "val"]["caption"])
    test_captions = set(cleaned_data[cleaned_data["split"] == "test"]["caption"])

    cleaned_data = data_train[~data_train["caption"].isin(val_captions | test_captions)]

    return cleaned_data


data = pd.read_csv("aligntune/data/RISCM/captions.csv")

data = data[["source", "split", "caption_1"]]

data = data.rename(columns={"caption_1": "caption"})
data_train_cleaned = remove_duplicates(data)

data_val_test = data[data["split"].isin(["val", "test"])]
data = pd.concat([data_train_cleaned, data_val_test], ignore_index=True)

data.to_csv("aligntune/data/RISCM/captions_cleaned.csv", index=False)
