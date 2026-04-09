"""Reports module for generating evaluation summaries."""

import pandas as pd
from typing import Optional
from pathlib import Path


def generate_summary_report(evaluator) -> pd.DataFrame:
    """Aggregate metrics across all entities."""
    overall = evaluator.get_overall_metrics()
    
    # Add model name if available
    if hasattr(evaluator, 'df') and 'name' in evaluator.df.columns:
        model_name = evaluator.df['name'].iloc[0] if len(evaluator.df) > 0 else 'Unknown'
        overall['model'] = model_name
    
    return overall


def generate_comparison_report(comparator) -> pd.DataFrame:
    """Multi-model comparison table with rankings."""
    comparison = comparator.get_comparison_table()
    ranking = comparator.get_ranking()
    
    # Combine comparison and ranking
    report = comparison.copy()
    report['best_model'] = comparison.idxmax(axis=1)
    
    # For CER, lower is better
    if 'cer_score' in report.index:
        report.loc['cer_score', 'best_model'] = comparison.loc['cer_score'].idxmin()
    
    return report


def generate_detailed_report(evaluator) -> dict:
    """Include per-entity, per-category, and null statistics."""
    return {
        'overall': evaluator.get_overall_metrics(),
        'per_entity': evaluator.get_per_entity_metrics(),
        'per_category': evaluator.get_per_category_metrics(),
        'null_statistics': evaluator.get_null_report(),
        'categories': evaluator.get_categories()
    }


def export_results(df: pd.DataFrame, path: str, format: str = 'csv'):
    """Save results to file."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'csv':
        df.to_csv(path, index=True)
    elif format == 'json':
        df.to_json(path, orient='records', indent=2)
    elif format == 'excel':
        df.to_excel(path, index=True)
    else:
        raise ValueError(f"Unsupported format: {format}")


def format_metrics_table(df: pd.DataFrame, precision: int = 3) -> str:
    """Pretty-print metrics for display."""
    return df.to_string(index=True, float_format=f'%.{precision}f')
