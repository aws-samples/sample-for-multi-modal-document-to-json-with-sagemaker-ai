import pandas as pd
from Levenshtein import distance


def unfold(row, text_propetry_name=None):
    return {
        "response_val": get_from_dict_safe(
            row["response"].get(row["entity"], None), text_propetry_name
        )[0],
        "label_val": get_from_dict_safe(
            row["labels"].get(row["entity"]), text_propetry_name
        )[0],
    }

def get_from_dict_safe(potential_dict, property_name) -> (str, bool):
    obj_type = type(potential_dict)
    if obj_type == dict and property_name in potential_dict:
        return (str(potential_dict[property_name]), True)
    return (str(potential_dict), False)


def add_edit_distance(df, columns, text_propetry_name):
    def calculate_edit_distance(row):
        response = row["response"]
        label = row["labels"]

        edit_distances = {}
        for key in columns:

            label_value = label.get(key, None)

            if response == {}:
                edit_distances[f"{key}"] = -4
            elif label_value == "None" or not label_value:
                edit_distances[f"{key}"] = -1
            elif key not in response:
                edit_distances[f"{key}"] = -3
            else:
                response_value = response[key]
                if response_value == "None" or not response_value:
                    edit_distances[f"{key}"] = -2
                else:
                    label_value, is_label_dict_format = get_from_dict_safe(
                        label_value, text_propetry_name
                    )
                    response_value, is_response_dict_format = get_from_dict_safe(
                        response_value, text_propetry_name
                    )

                    same_format = is_label_dict_format == is_response_dict_format
                    if same_format:
                        edit_distances[f"{key}"] = distance(response_value, label_value)
                    else:
                        edit_distances[f"{key}"] = -2

        return pd.Series(edit_distances)

    edit_distance_df = df.apply(calculate_edit_distance, axis=1)
    return pd.concat([df, edit_distance_df], axis=1)