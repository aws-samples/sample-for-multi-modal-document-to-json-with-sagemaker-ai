"""Feature categorization for evaluation."""

import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, Optional
from .config import CategorizationConfig


class FeatureCategory(Enum):
    """Feature type categories."""
    VALUE_EXTRACTION = "value_extraction"
    FREEFORM_TEXT = "freeform_text"
    CATEGORY_CLASSIFICATION = "category_classification"
    MISSING_GROUND_TRUTH = "missing_ground_truth"
    
    def __str__(self):
        return self.value


def is_mostly_null(values: pd.Series, threshold: float) -> bool:
    """Check if values are mostly null."""
    null_count = values.isna().sum() + values.isin(['', 'None', 'null', 'NaN', 'nan']).sum()
    null_ratio = null_count / len(values)
    return null_ratio > threshold


def is_constant_value(values: pd.Series) -> bool:
    """Check if all values are the same."""
    try:
        return values.nunique() == 1
    except TypeError:
        # Handle unhashable types (lists, dicts)
        # Convert to string for comparison
        return values.astype(str).nunique() == 1


def get_char_type_consistency(values: pd.Series) -> float:
    """Calculate character type pattern consistency."""
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
    
    if len(values) == 0:
        return 0.0
    
    patterns = values.apply(get_char_pattern)
    return patterns.value_counts().iloc[0] / len(patterns) if len(patterns) > 0 else 0.0


def get_structure_consistency(values: pd.Series) -> float:
    """Calculate structural pattern consistency."""
    def get_structure(s):
        return "".join(["W" if c.isalnum() else c for c in str(s)])
    
    if len(values) == 0:
        return 0.0
    
    structures = values.apply(get_structure)
    return structures.value_counts().iloc[0] / len(structures) if len(structures) > 0 else 0.0


def get_numeric_ratio(values: pd.Series) -> float:
    """Calculate ratio of numeric characters."""
    total_chars = sum(len(str(x)) for x in values)
    if total_chars == 0:
        return 0.0
    numeric_chars = sum(sum(c.isdigit() for c in str(x)) for x in values)
    return numeric_chars / total_chars


def get_special_char_ratio(values: pd.Series) -> float:
    """Calculate ratio of special characters."""
    total_chars = sum(len(str(x)) for x in values)
    if total_chars == 0:
        return 0.0
    special_chars = sum(sum(not c.isalnum() for c in str(x)) for x in values)
    return special_chars / total_chars


def is_value_extraction_statistical(values: pd.Series, config: CategorizationConfig) -> bool:
    """Determine if feature is value extraction based on statistical analysis."""
    values_str = values.astype(str)
    
    stats_features = {
        'length_std': values_str.str.len().std(),
        'length_mean': values_str.str.len().mean(),
        'unique_ratio': values.nunique() / len(values),
        'char_type_consistency': get_char_type_consistency(values_str),
        'structure_consistency': get_structure_consistency(values_str),
        'numeric_ratio': get_numeric_ratio(values_str),
        'special_char_ratio': get_special_char_ratio(values_str),
    }
    
    is_value_extraction = (
        (
            stats_features['length_std'] < config.length_std_threshold
            and stats_features['length_mean'] < config.length_mean_threshold
        )
        or (stats_features['char_type_consistency'] > config.char_type_consistency)
        or (stats_features['structure_consistency'] > config.structure_consistency)
        or (stats_features['numeric_ratio'] > config.numeric_ratio)
        or (
            stats_features['special_char_ratio'] > config.special_char_ratio_min
            and stats_features['special_char_ratio'] < config.special_char_ratio_max
        )
    )
    
    return is_value_extraction


def analyze_feature_distribution(df: pd.DataFrame, entity: str) -> Dict:
    """Statistical analysis of ground truth for a single entity."""
    values = df.apply(lambda row: row['labels'].get(entity), axis=1)
    
    # Convert string "None" to actual None/NaN
    values = values.replace(['None', 'null', 'NaN', 'nan', ''], None)
    
    # Basic statistics
    total_responses = len(values)
    unique_responses = values.astype(str).nunique()
    null_count = values.isnull().sum()
    
    # Length statistics (excluding nulls)
    clean_values = values.dropna()
    length_stats = clean_values.astype(str).str.len().describe() if len(clean_values) > 0 else pd.Series()
    
    # Character type analysis
    char_type_consistency = get_char_type_consistency(clean_values) if len(clean_values) > 0 else 0.0
    structure_consistency = get_structure_consistency(clean_values) if len(clean_values) > 0 else 0.0
    numeric_ratio = get_numeric_ratio(clean_values) if len(clean_values) > 0 else 0.0
    special_char_ratio = get_special_char_ratio(clean_values) if len(clean_values) > 0 else 0.0
    
    return {
        'entity': entity,
        'total_responses': total_responses,
        'unique_responses': unique_responses,
        'null_count': null_count,
        'null_percentage': (null_count / total_responses * 100) if total_responses > 0 else 0,
        'length_mean': length_stats.get('mean', 0),
        'length_std': length_stats.get('std', 0),
        'length_min': length_stats.get('min', 0),
        'length_max': length_stats.get('max', 0),
        'char_type_consistency': char_type_consistency,
        'structure_consistency': structure_consistency,
        'numeric_ratio': numeric_ratio,
        'special_char_ratio': special_char_ratio,
        'unique_ratio': (unique_responses / len(clean_values)) if len(clean_values) > 0 else 0,
    }


def categorize_features(df: pd.DataFrame, config: Optional[CategorizationConfig] = None) -> Dict[str, FeatureCategory]:
    """Automatic categorization based on thresholds."""
    if config is None:
        config = CategorizationConfig()
    
    # Extract all entities
    entities = set()
    for _, row in df.iterrows():
        if isinstance(row['labels'], dict):
            entities.update(row['labels'].keys())
    
    feature_categories = {}
    
    for entity in entities:
        values = df.apply(lambda row: row['labels'].get(entity), axis=1)
        
        # Check for mostly null
        if is_mostly_null(values, config.null_ratio_threshold):
            feature_categories[entity] = FeatureCategory.MISSING_GROUND_TRUTH
            continue
        
        clean_values = values.dropna()
        clean_values = clean_values[~clean_values.isin(['', 'None', 'null', 'NaN', 'nan'])]
        
        # Check for insufficient clean data
        if len(clean_values) < len(values) * config.min_clean_data_ratio:
            feature_categories[entity] = FeatureCategory.MISSING_GROUND_TRUTH
            continue
        
        # Check for constant value
        if is_constant_value(clean_values):
            feature_categories[entity] = FeatureCategory.MISSING_GROUND_TRUTH
            continue
        
        # Convert to string for analysis (handles lists/dicts)
        clean_values_str = clean_values.astype(str)
        
        # Check for value extraction
        if is_value_extraction_statistical(clean_values_str, config):
            feature_categories[entity] = FeatureCategory.VALUE_EXTRACTION
        # Check for category classification
        elif clean_values_str.nunique() < len(clean_values_str) * config.category_unique_ratio:
            feature_categories[entity] = FeatureCategory.CATEGORY_CLASSIFICATION
        else:
            feature_categories[entity] = FeatureCategory.FREEFORM_TEXT
    
    return feature_categories


def override_categories(categories: Dict[str, FeatureCategory], overrides: Dict[str, str]) -> Dict[str, FeatureCategory]:
    """Manual category assignment."""
    updated = categories.copy()
    for entity, category_str in overrides.items():
        try:
            updated[entity] = FeatureCategory(category_str)
        except ValueError:
            print(f"Warning: Invalid category '{category_str}' for entity '{entity}'")
    return updated
