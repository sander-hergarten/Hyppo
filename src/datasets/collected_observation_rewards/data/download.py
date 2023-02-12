import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import numpy as np
import json

pipeline_options = PipelineOptions(
    temp_location="gs://coinrun-dataset", project="deplearn"
)
query = "SELECT observation, reward " "FROM dataset_run_1674346924403.Coinrun"

path = "/Users/sanderhergarten/Documents/programming/Hyppo/src/datasets/collected_observation_rewards/data/data"


class TupelizeDoFn(beam.DoFn):
    def process(self, element):
        ls = []
        for index, chunk in enumerate(element["observation"]):
            ls.append({"observation": chunk, "reward": element["reward"][index]})

        return ls


def chunks(episode):
    """Yield n chunks from episode."""
    n = 20
    return {key: list(np.array_split(value, n)) for key, value in episode.items()}


def filter_empty_chunks(element):
    """Filter out empty chunks."""
    intermediate = []

    for index, chunk in enumerate(element["observation"]):
        if all([not x for x in chunk]):
            continue

        intermediate.append(index)

    construct = lambda key: [element[key][i] for i in intermediate]

    return {key: construct(key) for key in element.keys()}


def filter_empty_observations(element):
    """Filter out empty observations."""
    intermediate = {key: [] for key in element.keys()}

    for index, chunk in enumerate(element["observation"]):
        if any([not x for x in chunk]) and len(element["observation"]) > 1:
            number_of_empty = len(np.where(chunk == "")[0])

            for key in element.keys():
                previous = intermediate[key][index - 1]
                current = element[key][index]

                fixed_chunk = (
                    list(previous)[-number_of_empty:] + list(current)[:-number_of_empty]
                )

                intermediate[key].append(fixed_chunk)

            continue

        for key in element.keys():
            intermediate[key].append(list(element[key][index]))

    return intermediate


def assure_same_length(element):
    """Assure that all chunks have the same length."""
    intermediate = {key: [] for key in element.keys()}

    for index, chunk in enumerate(element["observation"]):
        if len(chunk) > 50:
            if index == 0 and len(element["observation"]) > 1:
                for key, value in element.items():
                    fixed_chunk = value[index][:50]
                    intermediate[key].append(fixed_chunk)

                continue

            elif index > 0 and len(element["observation"]) > 1:
                for key, value in element.items():
                    fixed_chunk = value[index][-50:]
                    intermediate[key].append(fixed_chunk)

                continue

            elif index == 0 and len(element["observation"]) == 1:
                for key, value in element.items():
                    fixed_chunk = list(value[index])[:40] + list(value[index])[-10:]
                    intermediate[key].append(fixed_chunk)

                continue

        for key, value in element.items():
            intermediate[key].append(value[index])

    return intermediate


def write(values):

    with open(path + f"/trajectory_{values[0]}.json", "w") as outfile:
        json.dump(values, outfile)
    return [1, 2, 3, 4]


def key_generator():
    ind = 0
    while True:
        yield ind
        ind += 1


key = key_generator()


def run():
    with beam.Pipeline(options=pipeline_options) as p:
        k = (
            p
            | "QueryTable"
            >> beam.io.ReadFromBigQuery(query=query, use_standard_sql=True)
            | "SubdivideEpisodes" >> beam.Map(chunks)
            | "FilterEmptyChunks" >> beam.Map(filter_empty_chunks)
            | "FilterEmptyObservations" >> beam.Map(filter_empty_observations)
            | "AssureSameLength" >> beam.Map(assure_same_length)
            | "Tupleize" >> beam.ParDo(TupelizeDoFn())
            | "Key" >> beam.Map(lambda x: (next(key), x))
            | "Write" >> beam.Map(write)
        )


if __name__ == "__main__":
    run()
