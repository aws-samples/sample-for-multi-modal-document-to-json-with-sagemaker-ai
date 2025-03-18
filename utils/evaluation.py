import pandas as pd
import json
import re

# from tqdm import tqdm
from tqdm.auto import tqdm  # for notebooks

# Create new `pandas` methods which use `tqdm` progress
# (can use tqdm_gui, optional kwargs, etc.)
tqdm.pandas()


import regex  # Requires the "regex" module (not standard "re")

regex.DEFAULT_VERSION = regex.VERSION1
pattern = r'\{(?:[^{}"]|"(?:\\.|[^"\\])*"|(?R))*\}'
json_pattern = regex.compile(pattern)

def to_dict(text):
    d = json.loads(text)
    if type(d) == str:
        # TODO remove now that dataset generation is fixed
        try:
            d = json.loads(d)
        except:
            return d
    for k, v in d.items():
        if type(v) == "str":
            d[k] = bytes(text, "utf-8").decode("unicode_escape")
    return d


def find_and_parse_json(input_string):
    # Find JSON-like substrings
    # json_pattern = re.compile(r'\{[^{}]*\}')

    # json_matches = json_pattern.findall(input_string)

    # input_string = input_string.strip(
    #     '"'
    # )  # TODO remove now that dataset generation is fixed
    # try:
    #     input_string = input_string.encode().decode(
    #         "unicode_escape"
    #     )  # TODO remove now that dataset generation is fixed
    # except:
    #     print("Decoding error")
    json_matches = json_pattern.search(input_string)

    # parsed_results = []

    # for json_str in json_matches:
    if json_matches:
        json_str = json_matches.group()
        try:
            # Attempt to parse the JSON string
            parsed_json = json.loads(json_str)
            # parsed_json = json_repair.loads(input_string)
            for k, v in parsed_json.items():
                if type(v) == "str":
                    parsed_json[k] = bytes(v, "utf-8").decode("unicode_escape")
            return parsed_json
        except json.JSONDecodeError:

            pass
            # print(f"Failed to parse JSON: {json_str}")
        # parsed_results.append(parsed_json)
    return {}
    # if len(parsed_results)>0:
    #     return parsed_results[0]
    # else:
    #     return {}


def get_path_without_extension(path, prefix_to_remove):
    # Remove '../results' from the path
    cleaned_path = path.replace(prefix_to_remove, "").strip("/")

    # Split path and extension
    path_without_ext = cleaned_path.rsplit(".", 1)[0]

    return path_without_ext


def parse_json_safely(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Handle malformed JSON with regex
        json_match = json_pattern.search(text)
        if json_match:
            return json.loads(json_match.group())
        return {}


def process_run(run, file, results_dir, expected_num_rows=-1):

    df = pd.read_json(file, lines=True)

    # try:
    if expected_num_rows == -1:
        expected_num_rows = len(df)
    elif len(df) != expected_num_rows:
        print(
            f"[Warning] Num rows: {len(df)} expected {expected_num_rows}. Skipping {file}."
        )
        return None, expected_num_rows

    pretty_name = run.human_name
    file_id = df.reset_index()["index"]
    model_name = run.model
    model = get_path_without_extension(file, results_dir)
    raw_response = df["response"]
    parsed_response = raw_response.apply(find_and_parse_json)
    label = df["labels"].apply(to_dict)

    df["pretty_name"] = pretty_name
    df["file_path"] = file
    df["fileid"] = file_id
    df["model_name"] = model_name
    df["model"] = model
    df["response_raw"] = raw_response
    df["response"] = parsed_response
    df["labels"] = label

    return df, expected_num_rows
    # except Exception as e:
    #     print(f"Error processing {file}: {e}")
    #     return None, expected_num_rows