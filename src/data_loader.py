from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split


def load_data():
    ds = load_dataset(
        "sapienzanlp-course-materials/hw-mnlp-2026"
    )

    ds_train_df = ds["train"].to_pandas()
    ds_train, ds_dev = train_test_split(ds_train_df, test_size=0.1, random_state=42)

    return {
        "train": ds_train,
        "dev": ds_dev,
        "test": ds["test"],
        "blind": ds["blind"]
    }
