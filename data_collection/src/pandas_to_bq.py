from typing import Optional

from google.cloud import bigquery
import pandas as pd
import time


class BigQueryPandasUploader:
    def __init__(
        self, schema: list[bigquery.SchemaField], project_id: str, dataset: str
    ):
        self.client = bigquery.Client()
        self.job_config = bigquery.LoadJobConfig(schema=schema)

        self.project_id = project_id
        self.dataset = dataset

    def _generate_table_id(self):
        return f"dataset_run_{int(time.time() *1000)}"

    def upload(self, df: pd.DataFrame, table_id: Optional[str] = None):

        if not table_id:
            table_id = self._generate_table_id()

        job = self.client.load_table_from_dataframe(
            df,
            f"{self.project_id}.{self.dataset}.{table_id}",
            job_config=self.job_config,
        )

        job.result()
