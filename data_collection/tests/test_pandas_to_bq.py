from google.cloud import bigquery
from run_saver import StepSaver
from pandas_to_bq import BigQueryPandasUploader
from tqdm import tqdm
from test_run_saver import RandomTrajectoryFactory


def test_pandas_to_bq():

    # Setup
    schema = [
        bigquery.SchemaField(
            "action", bigquery.enums.SqlTypeNames.INTEGER, mode="REPEATED"
        ),
        bigquery.SchemaField(
            "observation", bigquery.enums.SqlTypeNames.STRING, mode="REPEATED"
        ),
        bigquery.SchemaField(
            "discount", bigquery.enums.SqlTypeNames.FLOAT, mode="REPEATED"
        ),
        bigquery.SchemaField(
            "step_type", bigquery.enums.SqlTypeNames.INTEGER, mode="REPEATED"
        ),
        bigquery.SchemaField(
            "next_step_type", bigquery.enums.SqlTypeNames.INTEGER, mode="REPEATED"
        ),
        bigquery.SchemaField(
            "reward", bigquery.enums.SqlTypeNames.FLOAT, mode="REPEATED"
        ),
    ]

    trajectory_factory = RandomTrajectoryFactory(
        max_episode_len=10, enable_short_episodes=False
    )

    uploader = BigQueryPandasUploader(
        schema=schema, project_id="deplearn", dataset="test"
    )

    run_saver = StepSaver(batch_size=20, extend_length=10)

    # Generate Runs

    for k in tqdm(range(10)):
        generated_trajectory = trajectory_factory.generate()
        run_saver.add_data_to_queue(generated_trajectory)

    df = run_saver.commit_episode()
    print(df)

    uploader.upload(df)


if __name__ == "__main__":
    test_pandas_to_bq()
