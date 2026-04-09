"""Core metrics computation for evaluation."""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from Levenshtein import distance as levenshtein_distance
from cer import calculate_cer
from rouge_score import rouge_scorer


def compute_exact_match(df: pd.DataFrame, entities: List[str]) -> pd.DataFrame:
    """Compute per-entity exact match accuracy."""
    results = []
    
    for entity in entities:
        matches = 0
        total = 0
        
        for _, row in df.iterrows():
            pred = row['response_parsed'].get(entity)
            label = row['labels'].get(entity)
            
            # Skip if label is None/null
            if label is None or label in ['', 'None', 'null']:
                continue
            
            total += 1
            if str(pred) == str(label):
                matches += 1
        
        accuracy = matches / total if total > 0 else 0
        results.append({
            'entity': entity,
            'exact_match': accuracy,
            'matches': matches,
            'total': total
        })
    
    return pd.DataFrame(results)


def compute_edit_distance(df: pd.DataFrame, entities: List[str], text_property_name: Optional[str] = None) -> pd.DataFrame:
    """Compute character-level edit distance per entity."""
    
    def get_from_dict_safe(potential_dict, property_name):
        """Extract value from dict or return as string."""
        if isinstance(potential_dict, dict) and property_name and property_name in potential_dict:
            return str(potential_dict[property_name]), True
        return str(potential_dict) if potential_dict is not None else "", False
    
    results = []
    
    for _, row in df.iterrows():
        response = row['response_parsed']
        label = row['labels']
        
        row_result = {'file_id': row.get('file_id', 0), 'name': row.get('name', '')}
        
        for entity in entities:
            label_value = label.get(entity, None)
            
            # Handle different null/missing cases
            if not response or response == {}:
                row_result[entity] = -4  # Empty response
            elif label_value is None or label_value in ['', 'None', 'null']:
                row_result[entity] = -1  # Missing ground truth
            elif entity not in response:
                row_result[entity] = -3  # Missing key in response
            else:
                response_value = response[entity]
                if response_value is None or response_value in ['', 'None', 'null']:
                    row_result[entity] = -2  # Null value in response
                else:
                    # Extract text values
                    label_str, is_label_dict = get_from_dict_safe(label_value, text_property_name)
                    response_str, is_response_dict = get_from_dict_safe(response_value, text_property_name)
                    
                    if is_label_dict == is_response_dict:
                        row_result[entity] = levenshtein_distance(response_str, label_str)
                    else:
                        row_result[entity] = -2  # Format mismatch
        
        results.append(row_result)
    
    return pd.DataFrame(results)


def compute_cer(df: pd.DataFrame, entities: List[str], text_property_name: Optional[str] = None) -> pd.DataFrame:
    """Compute Character Error Rate per entity."""
    results = []
    
    for entity in entities:
        predictions = []
        references = []
        
        for _, row in df.iterrows():
            pred = row['response_parsed'].get(entity)
            label = row['labels'].get(entity)
            
            # Skip if label is None/null
            if label is None or label in ['', 'None', 'null']:
                continue
            
            # Extract text if nested
            if isinstance(pred, dict) and text_property_name:
                pred = pred.get(text_property_name, '')
            if isinstance(label, dict) and text_property_name:
                label = label.get(text_property_name, '')
            
            predictions.append(str(pred) if pred else '')
            references.append(str(label))
        
        if predictions and references:
            cer_score = calculate_cer(references, predictions)
        else:
            cer_score = 1.0  # Maximum error if no valid data
        
        results.append({
            'entity': entity,
            'cer_score': cer_score,
            'num_samples': len(predictions)
        })
    
    return pd.DataFrame(results)


def compute_rouge(df: pd.DataFrame, entities: List[str], text_property_name: Optional[str] = None) -> pd.DataFrame:
    """Compute ROUGE scores per entity."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    results = []
    
    for entity in entities:
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': [], 'rougeLsum': []}
        
        for _, row in df.iterrows():
            pred = row['response_parsed'].get(entity)
            label = row['labels'].get(entity)
            
            # Skip if label is None/null
            if label is None or label in ['', 'None', 'null']:
                continue
            
            # Extract text if nested
            if isinstance(pred, dict) and text_property_name:
                pred = pred.get(text_property_name, '')
            if isinstance(label, dict) and text_property_name:
                label = label.get(text_property_name, '')
            
            pred_str = str(pred) if pred else ''
            label_str = str(label)
            
            if pred_str and label_str:
                scores = scorer.score(label_str, pred_str)
                for metric in rouge_scores:
                    rouge_scores[metric].append(scores[metric].fmeasure)
        
        # Average scores
        result = {'entity': entity}
        for metric, scores in rouge_scores.items():
            result[metric] = np.mean(scores) if scores else 0.0
        result['num_samples'] = len(rouge_scores['rouge1'])
        
        results.append(result)
    
    return pd.DataFrame(results)


def compute_null_statistics(df: pd.DataFrame, entities: List[str], model: Optional[str] = None) -> pd.DataFrame:
    """Track null/missing values in predictions and labels."""
    results = []
    
    for entity in entities:
        null_in_labels = 0
        null_in_predictions = 0
        missing_key_in_predictions = 0
        total = len(df)
        
        for _, row in df.iterrows():
            label = row['labels'].get(entity)
            pred = row['response_parsed'].get(entity)
            
            # Check label nulls
            if label is None or label in ['', 'None', 'null']:
                null_in_labels += 1
            
            # Check prediction nulls
            if entity not in row['response_parsed']:
                missing_key_in_predictions += 1
            elif pred is None or pred in ['', 'None', 'null']:
                null_in_predictions += 1
        
        row_data = {
            'entity': entity,
            'total_samples': total,
            'null_in_labels': null_in_labels,
            'null_in_labels_pct': null_in_labels / total * 100 if total > 0 else 0,
            'null_in_predictions': null_in_predictions,
            'null_in_predictions_pct': null_in_predictions / total * 100 if total > 0 else 0,
            'missing_key_in_predictions': missing_key_in_predictions,
            'missing_key_pct': missing_key_in_predictions / total * 100 if total > 0 else 0,
            'valid_samples': total - null_in_labels
        }
        if model is not None:
            row_data['model'] = model
        results.append(row_data)
    
    return pd.DataFrame(results)


def compute_all_metrics(df: pd.DataFrame, config) -> Dict[str, pd.DataFrame]:
    """Orchestrate all metric computations."""
    from .config import EvaluationConfig
    
    if not isinstance(config, EvaluationConfig):
        config = EvaluationConfig()
    
    # Extract all entities from labels
    entities = set()
    for _, row in df.iterrows():
        if isinstance(row['labels'], dict):
            entities.update(row['labels'].keys())
    entities = sorted(list(entities))
    
    results = {}
    
    # Always compute null statistics
    results['null_statistics'] = compute_null_statistics(df, entities)
    
    if config.compute_exact_match:
        results['exact_match'] = compute_exact_match(df, entities)
    
    if config.compute_edit_distance:
        results['edit_distance'] = compute_edit_distance(df, entities, config.text_property_name)
    
    if config.compute_cer:
        results['cer'] = compute_cer(df, entities, config.text_property_name)
    
    if config.compute_rouge:
        results['rouge'] = compute_rouge(df, entities, config.text_property_name)
    
    return results
