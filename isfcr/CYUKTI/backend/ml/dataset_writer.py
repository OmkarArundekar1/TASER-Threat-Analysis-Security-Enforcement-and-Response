from pathlib import Path
from dataclasses import asdict
import pandas as pd
from feature_schema import CampaignDatasetRecord

class DatasetWriter:
    def __init__(
        self,
        dataset_directory="ml/datasets"
    ):
        self.dataset_directory = Path(dataset_directory)

        self.dataset_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.csv_file = (
            self.dataset_directory /
            "campaign_dataset.csv"
        )

        self.parquet_file = (
            self.dataset_directory /
            "campaign_dataset.parquet"
        )

    def append(
        self,
        record: CampaignDatasetRecord
    ):
        row = pd.DataFrame(
            [asdict(record)]
        )

        if self.csv_file.exists():
            row.to_csv(
                self.csv_file,
                mode="a",
                header=False,
                index=False
            )

        else:

            row.to_csv(
                self.csv_file,
                index=False
            )

    def export_parquet(self):
        if not self.csv_file.exists():
            return
        dataframe = pd.read_csv(
            self.csv_file
        )
        dataframe.to_parquet(
            self.parquet_file,
            index=False
        )

    def load(self):
        if not self.csv_file.exists():
            return pd.DataFrame()
        return pd.read_csv(
            self.csv_file
        )

    def size(self):
        if not self.csv_file.exists():
            return 0
        dataframe = pd.read_csv(
            self.csv_file
        )
        return len(dataframe)

writer = DatasetWriter()