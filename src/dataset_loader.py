import os
import pandas as pd


class DatasetLoader:

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = None

    def load(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"Metadata file not found:\n{self.csv_path}"
            )

        self.data = pd.read_csv(self.csv_path)

        print(f"[INFO] Loaded {len(self.data)} records.")

        return self.data

    def summary(self):
        if self.data is None:
            self.load()

        print("\n========== DATASET SUMMARY ==========\n")

        print(self.data.info())

        print("\nFirst 5 Records:\n")
        print(self.data.head())

        print("\nTotal Images :", len(self.data))

        if "age" in self.data.columns:
            print("Average Age :", round(self.data["age"].mean(), 2))

        print("\n=====================================\n")

    def total_images(self):
        if self.data is None:
            self.load()

        return len(self.data)

    def columns(self):
        if self.data is None:
            self.load()

        return list(self.data.columns)

    def get_dataframe(self):
        if self.data is None:
            self.load()

        return self.data
