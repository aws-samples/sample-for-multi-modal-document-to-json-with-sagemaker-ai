import ipywidgets as widgets
from enum import Enum
from IPython.display import display, HTML
import io
from PIL import Image
import base64

def image_base64(im):
    if isinstance(im, str):
        im = get_thumbnail_pdf(im)
    with io.BytesIO() as buffer:
        im.save(buffer, "png")
        return base64.b64encode(buffer.getvalue()).decode()


def image_formatter(im):
    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'



def create_parameter_widgets():
    style = {"description_width": "300px"}  # Increase width for longer descriptions
    layout = widgets.Layout(width="600px")  # Make sliders wider

    params = {
        "null_ratio_threshold": {
            "widget": widgets.FloatSlider(
                value=0.5,
                min=0.0,
                max=1.0,
                step=0.05,
                description="Null Ratio Threshold",
                style=style,
                layout=layout,
            ),
            "description": "Maximum allowed ratio of null values before considering feature useless",
        },
        "min_clean_data_ratio": {
            "widget": widgets.FloatSlider(
                value=0.3,
                min=0.0,
                max=1.0,
                step=0.05,
                description="Minimum Clean Data Ratio",
                style=style,
                layout=layout,
            ),
            "description": "Minimum ratio of non-null values required",
        },
        "category_unique_ratio": {
            "widget": widgets.FloatSlider(
                value=0.05,
                min=0.0,
                max=1.0,
                step=0.01,
                description="Category Uniqueness Ratio",
                style=style,
                layout=layout,
            ),
            "description": "Maximum ratio of unique values for category classification",
        },
        "freeform_text_length": {
            "widget": widgets.IntSlider(
                value=20,
                min=1,
                max=100,
                step=1,
                description="Freeform Text Length",
                style=style,
                layout=layout,
            ),
            "description": "Minimum average length for freeform text classification",
        },
        "length_std_threshold": {
            "widget": widgets.FloatSlider(
                value=2.0,
                min=0.0,
                max=10.0,
                step=0.1,
                description="Length Standard Deviation",
                style=style,
                layout=layout,
            ),
            "description": "Maximum standard deviation of length for value extraction",
        },
        "length_mean_threshold": {
            "widget": widgets.IntSlider(
                value=30,
                min=1,
                max=100,
                step=1,
                description="Mean Length Threshold",
                style=style,
                layout=layout,
            ),
            "description": "Maximum mean length for value extraction",
        },
        "char_type_consistency": {
            "widget": widgets.FloatSlider(
                value=0.8,
                min=0.0,
                max=1.0,
                step=0.05,
                description="Character Type Consistency",
                style=style,
                layout=layout,
            ),
            "description": "Minimum character type consistency for value extraction",
        },
        "structure_consistency": {
            "widget": widgets.FloatSlider(
                value=0.7,
                min=0.0,
                max=1.0,
                step=0.05,
                description="Structure Consistency",
                style=style,
                layout=layout,
            ),
            "description": "Minimum structure consistency for value extraction",
        },
        "numeric_ratio": {
            "widget": widgets.FloatSlider(
                value=0.5,
                min=0.0,
                max=1.0,
                step=0.05,
                description="Numeric Character Ratio",
                style=style,
                layout=layout,
            ),
            "description": "Minimum ratio of numeric characters for value extraction",
        },
        "special_char_ratio_min": {
            "widget": widgets.FloatSlider(
                value=0.1,
                min=0.0,
                max=1.0,
                step=0.05,
                description="Min Special Char Ratio",
                style=style,
                layout=layout,
            ),
            "description": "Minimum ratio of special characters for value extraction",
        },
        "special_char_ratio_max": {
            "widget": widgets.FloatSlider(
                value=0.3,
                min=0.0,
                max=1.0,
                step=0.05,
                description="Max Special Char Ratio",
                style=style,
                layout=layout,
            ),
            "description": "Maximum ratio of special characters for value extraction",
        },
    }

    return params


def display_parameter_widgets(params):
    # params = create_parameter_widgets()

    # Custom CSS for better spacing and readability
    display(
        HTML(
            """
        <style>
            .widget-group { margin-bottom: 20px; }
            .param-description { 
                margin-left: 10px;
                color: #666;
                font-style: italic;
                margin-bottom: 10px;
            }
        </style>
    """
        )
    )

    # Display widgets with descriptions
    for param_name, param_info in params.items():
        display(HTML(f'<div class="widget-group">'))
        display(param_info["widget"])
        display(
            HTML(f'<div class="param-description">{param_info["description"]}</div>')
        )
        display(HTML("</div>"))

    # Return just the widgets for later use
    return {name: param_info["widget"] for name, param_info in params.items()}


class FeatureCategory(Enum):
    MISSING_GROUND_TRUTH = "not_enough_ground_truth"
    FREEFORM_TEXT = "freeform_text"
    CATEGORY_CLASSIFICATION = "category_classification"
    VALUE_EXTRACTION = "value_extraction"

    def __str__(self):
        return self.value


# Modified functions that use the parameters


def categorize_features(df, params):
    feature_categories = {}

    for entity in df["entity"].unique():
        values = df[df["entity"] == entity]["label_val"]

        if is_mostly_null(values, params["null_ratio_threshold"]):
            feature_categories[entity] = FeatureCategory.MISSING_GROUND_TRUTH
            continue

        clean_values = values.dropna()

        if len(clean_values) < len(values) * params["min_clean_data_ratio"]:
            feature_categories[entity] = FeatureCategory.MISSING_GROUND_TRUTH
            continue

        if is_constant_value(clean_values):
            feature_categories[entity] = FeatureCategory.MISSING_GROUND_TRUTH
            continue

        if is_value_extraction_statistical(clean_values, params):
            feature_categories[entity] = FeatureCategory.VALUE_EXTRACTION

        elif (
            clean_values.nunique() < len(clean_values) * params["category_unique_ratio"]
        ):
            feature_categories[entity] = FeatureCategory.CATEGORY_CLASSIFICATION

        elif clean_values.astype(str).str.len().mean() > params["freeform_text_length"]:
            feature_categories[entity] = FeatureCategory.FREEFORM_TEXT

        else:
            feature_categories[entity] = FeatureCategory.MISSING_GROUND_TRUTH

    return feature_categories


def is_mostly_null(values, threshold):
    null_count = (
        values.isna().sum() + values.isin(["", "None", "null", "NaN", "nan"]).sum()
    )
    null_ratio = null_count / len(values)
    return null_ratio > threshold


def is_value_extraction_statistical(values, params):
    values_str = values.astype(str)

    stats_features = {
        "length_std": values_str.str.len().std(),
        "length_mean": values_str.str.len().mean(),
        "unique_ratio": values.nunique() / len(values),
        "char_type_consistency": get_char_type_consistency(values_str),
        "structure_consistency": get_structure_consistency(values_str),
        "numeric_ratio": get_numeric_ratio(values_str),
        "special_char_ratio": get_special_char_ratio(values_str),
    }

    is_value_extraction = (
        (
            stats_features["length_std"] < params["length_std_threshold"]
            and stats_features["length_mean"] < params["length_mean_threshold"]
        )
        or (stats_features["char_type_consistency"] > params["char_type_consistency"])
        or (stats_features["structure_consistency"] > params["structure_consistency"])
        or (stats_features["numeric_ratio"] > params["numeric_ratio"])
        or (
            stats_features["special_char_ratio"] > params["special_char_ratio_min"]
            and stats_features["special_char_ratio"] < params["special_char_ratio_max"]
        )
    )

    return is_value_extraction


def is_constant_value(values):
    """
    Check if all values in the series are the same
    """
    return values.nunique() == 1


def analyze_value_distribution(df):
    """
    Analyze and print detailed statistics for each entity
    """
    analysis_results = {}

    for entity in df["entity"].unique():
        values = df[df["entity"] == entity]["label_val"]

        # Calculate null statistics
        null_ratio = values.isna().sum() / len(values)
        null_like_ratio = values.isin(["", "None", "null", "NaN", "nan"]).sum() / len(
            values
        )

        # Clean values for other statistics
        clean_values = values.dropna()

        # Calculate constant value statistics
        is_constant = (
            is_constant_value(clean_values) if len(clean_values) > 0 else False
        )
        constant_value = (
            clean_values.iloc[0] if is_constant and len(clean_values) > 0 else None
        )

        values_str = clean_values.astype(str)

        analysis = {
            "null_ratio": null_ratio,
            "null_like_ratio": null_like_ratio,
            "total_null_ratio": null_ratio + null_like_ratio,
            "is_constant": is_constant,
            "constant_value": constant_value,
            "unique_values": clean_values.nunique(),
            "length_stats": {
                "mean": values_str.str.len().mean() if len(clean_values) > 0 else 0,
                "std": values_str.str.len().std() if len(clean_values) > 0 else 0,
                "min": values_str.str.len().min() if len(clean_values) > 0 else 0,
                "max": values_str.str.len().max() if len(clean_values) > 0 else 0,
            },
            "char_type_consistency": (
                get_char_type_consistency(values_str) if len(clean_values) > 0 else 0
            ),
            "structure_consistency": (
                get_structure_consistency(values_str) if len(clean_values) > 0 else 0
            ),
            "numeric_ratio": (
                get_numeric_ratio(values_str) if len(clean_values) > 0 else 0
            ),
            "special_char_ratio": (
                get_special_char_ratio(values_str) if len(clean_values) > 0 else 0
            ),
            "unique_ratio": (
                clean_values.nunique() / len(clean_values)
                if len(clean_values) > 0
                else 0
            ),
        }

        analysis_results[entity] = analysis

    return pd.DataFrame.from_dict(analysis_results, orient="index")


def get_char_type_consistency(values):
    def get_char_pattern(s):
        pattern = ""
        for c in str(s):
            if c.isdigit():
                pattern += "d"
            elif c.isalpha():
                pattern += "a"
            else:
                pattern += "s"
        return pattern

    patterns = values.apply(get_char_pattern)
    return patterns.value_counts().iloc[0] / len(patterns)


def get_structure_consistency(values):
    def get_structure(s):
        return "".join(["W" if c.isalnum() else c for c in str(s)])

    structures = values.apply(get_structure)
    return structures.value_counts().iloc[0] / len(structures)


def get_numeric_ratio(values):
    total_chars = sum(len(str(x)) for x in values)
    numeric_chars = sum(sum(c.isdigit() for c in str(x)) for x in values)
    return numeric_chars / total_chars if total_chars > 0 else 0


def get_special_char_ratio(values):
    total_chars = sum(len(str(x)) for x in values)
    special_chars = sum(sum(not c.isalnum() for c in str(x)) for x in values)
    return special_chars / total_chars if total_chars > 0 else 0


# Example usage:
# df = pd.DataFrame({
#     'entity': ['product_id', 'product_id', 'description', 'description'],
#     'label_val': ['ABC-123', 'DEF-456', 'Long text here', 'Another text']
# })
# categories = categorize_features(df)
# analysis = analyze_value_distribution(df)



import pandas as pd
import numpy as np
from itables import init_notebook_mode, show, options
import matplotlib.pyplot as plt
import math

# Initialize itables
init_notebook_mode(all_interactive=True)


def create_categorical_distribution(data, column_name):
    """Create a distribution plot for categorical data"""
    # Get value counts
    value_counts = data.value_counts().nlargest(10)

    plt.figure(figsize=(4, 3))
    plt.bar(
        range(len(value_counts)), value_counts.values, color="pink", edgecolor="black"
    )

    # Remove all axis labels and ticks
    plt.xticks([])
    plt.yticks([])
    plt.box(False)  # Remove the box around the plot

    plt.title(column_name, fontsize=10, pad=2)
    plt.tight_layout()

    # safe_filename = "".join(c for c in column_name if c.isalnum() or c in (' ', '_', '-'))

    # Save to a temporary file
    # plt.savefig(f'{safe_filename}_dist.png',
    #             bbox_inches='tight',
    #             transparent=True,
    #             dpi=100)
    # plt.close()

    # return f'{safe_filename}_dist.png'

    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg")
    buf.seek(0)
    pil_img = Image.open(buf)
    plt.close()
    return pil_img


def display_dataframe_with_distributions(df, plots_per_row=4):
    """Display DataFrame with categorical distribution plots in multiple rows"""

    # Filter categorical columns
    cat_columns = [
        col
        for col in df.columns
        if df[col].dtype == "object" or df[col].dtype.name == "category"
    ]

    # Calculate number of rows needed
    num_plots = len(cat_columns)
    num_rows = math.ceil(num_plots / plots_per_row)

    # Create CSS for layout
    css = """
    <style>
        .dist-container {
            display: flex;
            flex-direction: column;
            gap: 20px;
            margin-bottom: 20px;
        }
        .dist-row {
            display: flex;
            justify-content: flex-start;
            gap: 20px;
        }
        .dist-cell {
            flex: 1;
            min-width: 250px;
            max-width: 300px;
            text-align: center;
        }
        .dist-cell img {
            width: 100%;
            height: auto;
        }
    </style>
    """

    # Create distribution plots HTML
    dist_html = '<div class="dist-container">'

    # Create plots row by row
    for row in range(num_rows):
        dist_html += '<div class="dist-row">'

        # Get columns for this row
        start_idx = row * plots_per_row
        end_idx = min((row + 1) * plots_per_row, num_plots)
        row_columns = cat_columns[start_idx:end_idx]

        # Add plots for this row
        for column in row_columns:
            img = create_categorical_distribution(df[column], column)
            dist_html += f"""
                <div class="dist-cell">
                    {image_formatter(img)}
                </div>
            """

        dist_html += "</div>"

    dist_html += "</div>"

    # Display distributions
    display(HTML(css + dist_html))

    # Display interactive table with itables
    df.style.hide(axis="index")
    options.showIndex = False
    show(df, scrollY="400px", classes=["display", "compact"], index=False)



def create_interactive_categorization(feature_categories):
    # Convert string values to enum values
    enum_categories = {
        key: FeatureCategory(value) for key, value in feature_categories.items()
    }

    dropdowns = {}
    for entity, current_category in enum_categories.items():
        dropdown = widgets.Dropdown(
            options=list(FeatureCategory),
            value=current_category,
            description=f"{entity}:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="300px"),
        )
        dropdowns[entity] = dropdown

    return dropdowns