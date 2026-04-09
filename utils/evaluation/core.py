"""Core evaluator classes."""

import pandas as pd
from typing import Optional, Dict
from .config import EvaluationConfig
from .metrics import compute_all_metrics
from .categorization import categorize_features, FeatureCategory


class SingleModelEvaluator:
    """Evaluate one model's results."""
    
    def __init__(self, df: pd.DataFrame, config: Optional[EvaluationConfig] = None):
        self.df = df
        self.config = config or EvaluationConfig()
        self._metrics = None
        self._categories = None
    
    def evaluate(self) -> pd.DataFrame:
        """Run full evaluation pipeline."""
        # Compute all metrics
        self._metrics = compute_all_metrics(self.df, self.config)
        
        # Categorize features
        self._categories = categorize_features(self.df)
        
        return self.get_overall_metrics()
    
    def get_overall_metrics(self) -> pd.DataFrame:
        """Aggregated metrics across all entities."""
        if self._metrics is None:
            self.evaluate()
        
        # Aggregate metrics across all entities
        results = []
        
        # Exact match
        if 'exact_match' in self._metrics:
            em_df = self._metrics['exact_match']
            results.append({
                'metric': 'exact_match',
                'value': em_df['exact_match'].mean(),
                'count': em_df['total'].sum()
            })
        
        # CER
        if 'cer' in self._metrics:
            cer_df = self._metrics['cer']
            results.append({
                'metric': 'cer_score',
                'value': cer_df['cer_score'].mean(),
                'count': cer_df['num_samples'].sum()
            })
        
        # ROUGE
        if 'rouge' in self._metrics:
            rouge_df = self._metrics['rouge']
            for metric in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']:
                if metric in rouge_df.columns:
                    results.append({
                        'metric': metric,
                        'value': rouge_df[metric].mean(),
                        'count': rouge_df['num_samples'].iloc[0] if len(rouge_df) > 0 else 0
                    })
        
        return pd.DataFrame(results)
    
    def get_per_entity_metrics(self) -> pd.DataFrame:
        """Metrics broken down by entity."""
        if self._metrics is None:
            self.evaluate()
        
        # Merge all metrics by entity
        result_df = None
        
        for metric_name, metric_df in self._metrics.items():
            if metric_name == 'edit_distance':
                continue  # Skip edit distance (per-sample metric)
            
            if 'entity' in metric_df.columns:
                if result_df is None:
                    result_df = metric_df.copy()
                else:
                    result_df = result_df.merge(metric_df, on='entity', how='outer', suffixes=('', f'_{metric_name}'))
        
        return result_df if result_df is not None else pd.DataFrame()
    
    def get_per_category_metrics(self) -> pd.DataFrame:
        """Metrics grouped by feature category."""
        if self._metrics is None or self._categories is None:
            self.evaluate()
        
        per_entity = self.get_per_entity_metrics()
        if per_entity.empty:
            return pd.DataFrame()
        
        # Add category column
        per_entity['category'] = per_entity['entity'].map(
            lambda e: str(self._categories.get(e, FeatureCategory.MISSING_GROUND_TRUTH))
        )
        
        # Group by category and aggregate
        numeric_cols = per_entity.select_dtypes(include=['number']).columns
        grouped = per_entity.groupby('category')[numeric_cols].mean().reset_index()
        
        return grouped
    
    def get_null_report(self) -> pd.DataFrame:
        """Null value statistics."""
        if self._metrics is None:
            self.evaluate()
        
        return self._metrics.get('null_statistics', pd.DataFrame())
    
    def get_categories(self) -> Dict[str, FeatureCategory]:
        """Get feature categories."""
        if self._categories is None:
            self._categories = categorize_features(self.df)
        return self._categories


class MultiModelComparator:
    """Compare multiple models."""
    
    def __init__(self, results_dict: Dict[str, pd.DataFrame], config: Optional[EvaluationConfig] = None):
        self.results_dict = results_dict
        self.config = config or EvaluationConfig()
        self._evaluators = {}
        self._comparison = None
    
    def compare(self) -> pd.DataFrame:
        """Run comparison across all models."""
        # Create evaluators for each model
        for name, df in self.results_dict.items():
            self._evaluators[name] = SingleModelEvaluator(df, self.config)
            self._evaluators[name].evaluate()
        
        # Build comparison table
        self._comparison = self.get_comparison_table()
        return self._comparison
    
    def get_comparison_table(self) -> pd.DataFrame:
        """Side-by-side comparison DataFrame."""
        all_results = []
        
        for name, evaluator in self._evaluators.items():
            overall = evaluator.get_overall_metrics()
            overall['model'] = name
            all_results.append(overall)
        
        if not all_results:
            return pd.DataFrame()
        
        # Combine and pivot
        combined = pd.concat(all_results, ignore_index=True)
        pivot = combined.pivot(index='metric', columns='model', values='value')
        
        return pivot
    
    def get_best_model(self, metric: str) -> str:
        """Identify best performing model."""
        if self._comparison is None:
            self.compare()
        
        if metric not in self._comparison.index:
            raise ValueError(f"Metric '{metric}' not found in comparison")
        
        # For CER, lower is better; for others, higher is better
        if metric == 'cer_score':
            return self._comparison.loc[metric].idxmin()
        else:
            return self._comparison.loc[metric].idxmax()
    
    def get_ranking(self) -> pd.DataFrame:
        """Rank models by multiple metrics."""
        if self._comparison is None:
            self.compare()
        
        rankings = []
        
        for metric in self._comparison.index:
            # For CER, lower is better (rank ascending)
            # For others, higher is better (rank descending)
            ascending = (metric == 'cer_score')
            ranked = self._comparison.loc[metric].rank(ascending=ascending, method='min')
            
            rankings.append({
                'metric': metric,
                **{model: int(rank) for model, rank in ranked.items()}
            })
        
        return pd.DataFrame(rankings)
