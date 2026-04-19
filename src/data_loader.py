from datasets import Dataset, load_dataset
#from sklearn.model_selection import train_test_split


def load_data():
    ds = load_dataset(
        "sapienzanlp-course-materials/hw-mnlp-2026"
    )

    ds_train = ds["train"]
    train_dev_split = ds["train"].train_test_split(test_size=0.1, seed=42)
    ds_train = train_dev_split["train"]
    ds_dev = train_dev_split["test"]

    return {
        "train": ds_train,
        "dev": ds_dev,
        "test": ds["test"],
        "blind": ds["blind"]
    }
