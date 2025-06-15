import pandas as pd

df = pd.read_csv("aligntune/data/RISCM/captions_cleaned.csv")


train_captions = df[df["split"] == "train"]["caption"]
val_captions = df[df["split"] == "val"]["caption"]


common_captions = pd.Series(list(set(train_captions) & set(val_captions)))
assert (
    len(common_captions) == 0
), "There are common captions between train and val splits!"
print(f"Number of common captions: {len(common_captions)}")
