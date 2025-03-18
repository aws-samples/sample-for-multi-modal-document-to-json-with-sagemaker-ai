import os
import glob
import random
import base64
import pandas as pd

from PIL import Image
from io import BytesIO
from IPython.display import HTML

# pd.set_option('display.max_colwidth', 0)
import PIL
import io
import difflib
import sys



def get_thumbnail(path):
    i = Image.open(path)
    i.thumbnail((150, 150), Image.LANCZOS)
    return i


def image_base64(im):
    if isinstance(im, str):
        im = get_thumbnail(im)
    with BytesIO() as buffer:
        im.save(buffer, "png")
        return base64.b64encode(buffer.getvalue()).decode()


def image_formatter(im):
    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'


def get_pil_image(im, dataset_base_dir=""):
    if type(im) is dict:
        im_bytes = im.get("bytes", None)
        if im_bytes:
            im = BytesIO(im_bytes)
        else:
            im = os.path.join(dataset_base_dir, im.get("path", None))

    image = PIL.Image.open(im)
    return image.resize((400, 600))

def get_dict_str_aligned(first_dict, second_dict):
    keys = set(list(first_dict.keys()) + list(second_dict.keys()))

    first_str = []
    second_str = []
    for key in keys:
        first_str.append(f"{key}: {first_dict.get(key, '')}")
        second_str.append(f"{key}: {second_dict.get(key, '')}")

    return first_str, second_str



def get_diff(row):

    fromlines, tolines = get_dict_str_aligned(row.labels, row.response)

    diff = difflib.HtmlDiff().make_file(fromlines, tolines, "labels", "response")
    legend_prefix = '<table class="diff" summary="Legends">'
    legend_suffix = "</table>\n</body>"
    legend_start = diff.find(legend_prefix)
    legend_end = diff.find(legend_suffix, legend_start + len(legend_prefix))

    diff_without_legend = diff[:legend_start] + diff[legend_end + len("</table>") :]
    # display(HTML(diff_without_legend))

    return diff_without_legend.replace("\n", " ")


# example usage:
# row = df_final.iloc[7]
# diff = get_diff(row)
# display(HTML(diff))