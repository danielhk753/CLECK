import os

import ujson as json
from tqdm import tqdm


# Define datasets paths:
DATASETS = {
    'TriviaQA': "C:/Users/danie/Desktop/triviaqa-unfiltered/triviaqa-unfiltered/unfiltered-web-train.json",
    'Zero-Shot RE': None,
    'SQuAD': None,
    'Natural Questions': None,
    'MLQA': None,
    'T-REx': None
}


def get_json_file(file_path, dataset_name, chunk_size_mb=10):
    # Define the chunk size (in bytes)
    chunk_size = 1024 * 1024 * chunk_size_mb
    print('Setting chunk size')

    # Get file size for progress tracking
    file_size = os.path.getsize(file_path)
    print(f'File size: {file_size}')

    # Read and parse JSON with a progress bar
    json_data = ""
    with open(file_path, "r", encoding="utf-8") as file, tqdm(
            total=file_size, unit="B", unit_scale=True, desc=f"Loading {dataset_name} JSON file"
    ) as pbar:
        while chunk := file.read(chunk_size):
            json_data += chunk
            pbar.update(len(chunk.encode("utf-8")))  # Update progress based on bytes read

    # Parse JSON after reading
    data = json.loads(json_data)

    return data


def get_trivia_qa(path):
    # Load dataset from path:
    data = get_json_file(path, 'TriviaQA')

    # Extract QA pairs:
    dataset = []
    for sample in tqdm(data['Data'], desc='Extracting TriviaQA dataset QA pairs'):

        question = sample['Question']
        answer_type = sample['Answer']['Type']

        if answer_type == 'WikipediaEntity':
            answer = sample['Answer']['MatchedWikiEntityName']
        else:
            continue

        dataset.append({'Question': question,
                        'Answer': answer})

    return dataset


def get_zero_shot_re(path):
    pass


def get_squad(path):
    pass


def get_nq(path):
    pass


def get_mlqa(path):
    pass


def get_trex(path):
    pass


def get_datasets(datasets=DATASETS):
    dataset = []
    num_datasets = 0

    if datasets['TriviaQA']:
        dataset += get_trivia_qa(datasets['TriviaQA'])
        num_datasets += 1
    if datasets['Zero-Shot RE']:
        dataset += get_zero_shot_re(datasets['Zero-Shot RE'])
        num_datasets += 1
    if datasets['SQuAD']:
        dataset += get_squad(datasets['SQuAD'])
        num_datasets += 1
    if datasets['Natural Questions']:
        dataset += get_nq(datasets['Natural Questions'])
        num_datasets += 1
    if datasets['MLQA']:
        dataset += get_mlqa(datasets['MLQA'])
        num_datasets += 1
    if datasets['T-REx']:
        dataset += get_trex(datasets['T-REx'])
        num_datasets += 1

    print(f"Collected {len(dataset)} QA pairs from {num_datasets} available datasets.")

    return dataset
