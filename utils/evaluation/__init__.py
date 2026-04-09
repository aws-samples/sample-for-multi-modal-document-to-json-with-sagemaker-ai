"""
Evaluation package for document-to-JSON model assessment.

This package provides a clean API for evaluating vision LLM models on document understanding tasks.
"""

# Lazy imports to avoid dependency issues during package initialization
def __getattr__(name):
    if name in ("EvaluationConfig", "CategorizationConfig", "VisualizationConfig", "ResultSource"):
        from .config import EvaluationConfig, CategorizationConfig, VisualizationConfig, ResultSource
        return locals()[name]
    elif name in ("SingleModelEvaluator", "MultiModelComparator"):
        from .core import SingleModelEvaluator, MultiModelComparator
        return locals()[name]
    elif name == "ResultsRegistry":
        from .loaders import ResultsRegistry
        return ResultsRegistry
    elif name == "FeatureCategory":
        from .categorization import FeatureCategory
        return FeatureCategory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Convenience functions
def load_results(csv_path: str, dataset_base_dir: str = ""):
    """Load results from CSV file.
    
    Args:
        csv_path: Path to CSV file with columns: name, location, model_name?
        dataset_base_dir: Base directory for resolving image paths
    
    Returns:
        DataFrame with loaded results
    """
    from .loaders import ResultsRegistry
    registry = ResultsRegistry()
    registry.add_from_csv(csv_path, dataset_base_dir)
    return registry.load_all()


def evaluate_model(df, config=None):
    """Evaluate a single model.
    
    Args:
        df: DataFrame with response_parsed and labels columns
        config: Optional EvaluationConfig
    
    Returns:
        SingleModelEvaluator instance
    """
    from .core import SingleModelEvaluator
    evaluator = SingleModelEvaluator(df, config)
    evaluator.evaluate()
    return evaluator


def compare_models(results_dict, config=None):
    """Compare multiple models.
    
    Args:
        results_dict: Dict mapping model names to DataFrames
        config: Optional EvaluationConfig
    
    Returns:
        MultiModelComparator instance
    """
    from .core import MultiModelComparator
    comparator = MultiModelComparator(results_dict, config)
    comparator.compare()
    return comparator


__all__ = [
    # Config classes
    "EvaluationConfig",
    "CategorizationConfig",
    "VisualizationConfig",
    "ResultSource",
    # Core classes
    "SingleModelEvaluator",
    "MultiModelComparator",
    "ResultsRegistry",
    "FeatureCategory",
    # Convenience functions
    "load_results",
    "evaluate_model",
    "compare_models",
]
