"""Configuration dataclasses for evaluation."""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ResultSource:
    """Unified representation of inference results."""
    name: str
    location: str
    model_name: Optional[str] = None


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics and processing."""
    
    # Metric selection
    compute_exact_match: bool = True
    compute_edit_distance: bool = True
    compute_cer: bool = True
    compute_rouge: bool = True
    
    # Text property handling
    text_property_name: Optional[str] = None
    
    # Null handling
    null_values: List[str] = field(default_factory=lambda: ["", "None", "null", "NaN", "nan"])
    
    # Dataset configuration
    dataset_base_dir: str = ""


@dataclass
class CategorizationConfig:
    """Configuration for feature categorization thresholds."""
    
    # Null ratio thresholds
    null_ratio_threshold: float = 0.7
    min_clean_data_ratio: float = 0.3
    
    # Category classification
    category_unique_ratio: float = 0.05
    
    # Freeform text
    freeform_text_length: int = 20
    
    # Value extraction
    length_std_threshold: float = 2.0
    length_mean_threshold: int = 30
    char_type_consistency: float = 0.8
    structure_consistency: float = 0.7
    numeric_ratio: float = 0.5
    special_char_ratio_min: float = 0.1
    special_char_ratio_max: float = 0.3


@dataclass
class VisualizationConfig:
    """Configuration for plot styling."""
    
    figsize: tuple = (10, 6)
    dpi: int = 100
    style: str = "seaborn-v0_8-darkgrid"
    color_palette: str = "Set2"
    font_size: int = 10
