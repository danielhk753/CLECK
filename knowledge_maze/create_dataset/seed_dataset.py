import os
import json

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def load_json(file_path):
    """
    Loads and parses JSON data from a file.

    This function reads a JSON file from the provided file path and returns its
    contents as a Python object. If the file path is not provided or the file does
    not exist, an empty list is returned.

    Args:
        file_path (str): The path to the JSON file to be loaded.

    Returns:
        Any: The Python object resulting from the parsed JSON file. If the file
        does not exist or the path is not provided, an empty list is returned.
    """
    if not file_path or not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


# Get dataset file paths from environment variables
fanoutqa_dev_path = os.getenv("FANOUTQA_DEV_PATH")
fanoutqa_test_path = os.getenv("FANOUTQA_TEST_PATH")

# Load datasets
fanoutqa_dev = load_json(fanoutqa_dev_path)
fanoutqa_test = load_json(fanoutqa_test_path)


def extract_questions(data_sample):
    """
    Extracts all questions from a nested data structure.

    The function traverses a nested data structure, which can be a combination of
    dictionaries and lists, and retrieves all values associated with keys named
    "question". It supports deeply nested structures by recursively examining each
    level of the data.

    Args:
        data_sample (dict | list): The input data structure containing potential
            "question" keys. It can be a dictionary or a list of dictionaries/lists.

    Returns:
        list: A list of strings containing all the values associated with
            "question" keys found within the input data structure.
    """
    questions = []
    if isinstance(data_sample, dict):
        temp = {}
        for key, value in data_sample.items():
            if key == "question":
                temp['sub-question'] = value
            if key == "answer":
                temp['sub-answer'] = value
            if temp not in questions and temp:
                questions.append(temp)
            questions.extend(extract_questions(value))
    elif isinstance(data_sample, list):
        for item in data_sample:
            questions.extend(extract_questions(item))
    return questions


def transform_data(dataset):
    """
    Transforms a dataset by extracting specific fields and processing structured
    components like decompositions and URLs. Each item in the dataset is processed
    to construct a transformed representation with fields including 'id',
    'question', 'answer', and 'urls'. It handles nested dictionaries and lists,
    ensuring only valid URLs are included. Errors during processing are logged,
    and problematic items are skipped to ensure robust transformation.

    Args:
        dataset (list[dict]): The input list of dictionaries where each dictionary
            represents an item with fields such as 'id', 'question', 'answer', and
            'decomposition'. The 'decomposition' is expected to be a list of
            dictionaries, each optionally containing an 'evidence' dictionary with
            'url' field.

    Returns:
        list[dict]: A list containing transformed dictionaries with the following
            fields:
            - 'id': The 'id' of the item, defaulting to an empty string if absent.
            - 'question': The 'question' field from the item, defaulting to an
              empty string if absent.
            - 'answer': The 'answer' field from the item as-is, defaulting to an
              empty dictionary if absent.
            - 'urls': A list of non-empty URLs extracted from the 'decomposition'
              in the input item. Defaults to an empty list if no URLs are present
              or the 'decomposition' is invalid.
    """
    transformed_data = []

    for item in dataset or []:
        try:
            urls = []
            decompositions = item.get("decomposition", [])

            # Check if decompositions is a list and not None
            if decompositions and isinstance(decompositions, list):
                for decomposition in decompositions:
                    # Check if decomposition is a dictionary and not None
                    if decomposition and isinstance(decomposition, dict):
                        evidence = decomposition.get("evidence", {})
                        # Check if evidence is a dictionary and not None
                        if evidence and isinstance(evidence, dict):
                            url = evidence.get("url", "")
                            if url:  # Only add non-empty URLs
                                urls.append(url)

            transformed_item = {
                "id": item.get("id", ""),
                "question": item.get("question", ""),
                "answer": item.get("answer", {}),
                "urls": urls,
                "sub-questions": extract_questions(item)[1:]
            }
            if isinstance(item.get("answer", {}), dict):
                transformed_data.append(transformed_item)
        except Exception as e:
            print(f"Error processing item: {e}")
            print(f"Problematic item: {item}")
            # Continue with the next item instead of failing

    return transformed_data


def get_combined_transformed_datasets():
    """
    Combines and transforms datasets for further processing.

    This function aggregates and processes datasets from two specified
    sources: `fanoutqa_dev` and `fanoutqa_test`. It achieves this by
    applying a transformation function to each dataset individually
    and subsequently combining their results into a single output.

    Returns:
        list: The combined list resulting from the individual transformations
        of the `fanoutqa_dev` and `fanoutqa_test` datasets.
    """
    return transform_data(fanoutqa_dev) + transform_data(fanoutqa_test)
