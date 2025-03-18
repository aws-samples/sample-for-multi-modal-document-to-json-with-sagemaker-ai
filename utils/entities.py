import pandas as pd
import numpy as np
from collections import Counter
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_string_distribution_by_entity(df_multi, col_value="label_val"):
    # Initialize list to store results
    results = []
    
    # Get unique entities
    entities = df_multi['entity'].unique()
    
    for entity in entities:
        # Filter dataframe for current entity
        df_entity = df_multi[df_multi['entity'] == entity].copy()
        
        # Convert string "None" to actual None/NaN
        df_entity.loc[df_entity[col_value].astype(str).str.lower() == 'none', col_value] = None
        
        # 1. Basic statistics
        total_responses = len(df_entity)
        unique_responses = df_entity[col_value].nunique()
        null_count = df_entity[col_value].isnull().sum()
        
        results.extend([
            {'entity': entity, 
             'metric_name': 'total_responses',
             'metric_description': 'Total number of responses',
             'metric_value': total_responses},
            {'entity': entity, 
             'metric_name': 'unique_responses',
             'metric_description': 'Number of unique responses',
             'metric_value': unique_responses},
            {'entity': entity, 
             'metric_name': 'null_count',
             'metric_description': 'Number of null values (including "None")',
             'metric_value': null_count}
        ])
        
        # 2. Length statistics (excluding nulls)
        length_stats = df_entity[col_value].dropna().astype(str).str.len().describe()
        for stat_name in ['mean', 'std']: #, 'min', '25%', '50%', '75%', 'max'
            results.append({
                'entity': entity,
                'metric_name': f'length_{stat_name}',
                'metric_description': f'Response length {stat_name} (excluding nulls)',
                'metric_value': length_stats[stat_name]
            })
        
        # 3. Most common responses (excluding nulls)
        top_responses = df_entity[col_value].dropna().value_counts().head(10)
        results.append({
            'entity': entity,
            'metric_name': 'top_10_responses',
            'metric_description': 'Top 10 most common responses (excluding nulls)',
            'metric_value': top_responses.to_dict()
        })
        
        # 4. Numeric content (excluding nulls)
        def contains_number(x):
            return any(char.isdigit() for char in str(x))
        
        numeric_responses = df_entity[col_value].dropna().apply(contains_number).sum()
        valid_responses = len(df_entity[col_value].dropna())
        results.append({
            'entity': entity,
            'metric_name': 'numeric_responses',
            'metric_description': 'Number of responses containing numeric values (excluding nulls)',
            'metric_value': numeric_responses
        })
        
        # 5. Word frequency (excluding nulls)
        def get_words(x):
            return str(x).lower().split()
        
        all_words = df_entity[col_value].dropna().apply(get_words)
        word_freq = Counter([word for sublist in all_words for word in sublist])
        top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10])
        
        results.append({
            'entity': entity,
            'metric_name': 'top_10_words',
            'metric_description': 'Top 10 most frequent words (excluding nulls)',
            'metric_value': top_words
        })
        
        # 6. Response patterns (excluding nulls)
        def categorize_response(x):
            x = str(x)            
            if x.isdigit():
                return 'numeric'
            elif contains_number(x):
                return 'mixed'
            elif len(x.split()) > 10:
                return 'long_text'
            else:
                return 'short_text'
        
        response_types = df_entity[col_value].dropna().apply(categorize_response).value_counts()
        
        # Add null count to response types
        response_types_dict = response_types.to_dict()
        response_types_dict['null'] = null_count
        
        results.append({
            'entity': entity,
            'metric_name': 'response_types',
            'metric_description': 'Distribution of response types (including nulls)',
            'metric_value': response_types_dict
        })
        
        # 7. Add null percentage
        results.append({
            'entity': entity,
            'metric_name': 'null_percentage',
            'metric_description': 'Percentage of null values (including "None")',
            'metric_value': (null_count / total_responses * 100) if total_responses > 0 else 0
        })
    
    # Convert results to dataframe
    results_df = pd.DataFrame(results)
    return results_df


def plot_length_boxplots_with_limits_horizontal(df_multi, max_length=None, log_scale=True, col_value="label_value"):
    """
    Create horizontal boxplots with optional maximum length limit and log scale
    
    Parameters:
    -----------
    df_multi : pandas DataFrame
        Input DataFrame containing 'entity' and 'col_value' columns
    max_length : int, optional
        Maximum length to include in the analysis
    log_scale : bool, default=True
        Whether to use log scale for the length axis
    """
    # Create a copy and convert string "None" to actual None/NaN
    df = df_multi.copy()
    df.loc[df[col_value].astype(str).str.lower() == 'none', col_value] = None
    
    # Calculate lengths (excluding nulls)
    df['length'] = df[col_value].dropna().astype(str).str.len()
    
    # Store original counts for statistics
    original_counts = df.groupby('entity')['length'].count()
    
    # Apply max length filter if specified
    if max_length is not None:
        df_filtered = df[df['length'] <= max_length]
        filtered_counts = df_filtered.groupby('entity')['length'].count()
        truncated_counts = original_counts - filtered_counts
    else:
        df_filtered = df
        truncated_counts = pd.Series(0, index=df['entity'].unique())
    
    # Remove outliers for each entity separately
    df_no_outliers = pd.DataFrame()
    for entity in df_filtered['entity'].unique():
        entity_data = df_filtered[df_filtered['entity'] == entity]
        clean_data = remove_outliers(entity_data, 'length')
        df_no_outliers = pd.concat([df_no_outliers, clean_data])
    
    # Create figure with two subplots side by side
    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(2, 1, width_ratios=[1])
    
    # Create main plot
    ax1 = fig.add_subplot(gs[0])
    
    # Create boxplot
    sns.boxplot(data=df_no_outliers, y='entity', x='length', color='skyblue', ax=ax1)
    sns.stripplot(data=df_no_outliers, y='entity', x='length', 
                 color='navy', alpha=0.2, size=4, jitter=0.2, ax=ax1)
    
    # Set log scale if requested
    if log_scale:
        ax1.set_xscale('log')
    
    # Customize the boxplot
    title = 'Response Length Distribution by Entity (Outliers Removed)'
    if max_length is not None:
        title += f'\nMax length: {max_length}'
    ax1.set_title(title, fontsize=12, pad=20)
    ax1.set_ylabel('Entity', fontsize=10)
    ax1.set_xlabel('Length (log scale)' if log_scale else 'Length', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Calculate statistics for each entity
    stats_original = df.groupby('entity')['length'].agg([
        ('total_count', 'count'),
        ('null_count', lambda x: x.isnull().sum())
    ])
    
    stats_clean = df_no_outliers.groupby('entity')['length'].agg([
        ('mean', lambda x: f"{x.mean():.1f}"),
        ('median', lambda x: f"{x.median():.1f}"),
        ('std', lambda x: f"{x.std():.1f}"),
        ('min', lambda x: f"{x.min():.1f}"),
        ('max', lambda x: f"{x.max():.1f}"),
        ('clean_count', 'count')
    ])
    
    # Calculate outliers and truncated counts
    stats_combined = stats_original.join(stats_clean)
    stats_combined['truncated'] = truncated_counts
    stats_combined['outliers_removed'] = (stats_combined['total_count'] - 
                                        stats_combined['truncated'] - 
                                        stats_combined['clean_count'])
    
    # Reorder columns
    stats_combined = stats_combined[[
        'total_count', 'null_count', 'truncated', 'outliers_removed', 'clean_count',
        'mean', 'median', 'std', 'min', 'max'
    ]]
    
    # Create table subplot
    ax2 = fig.add_subplot(gs[1])
    
    # Create a table with statistics
    table = ax2.table(cellText=stats_combined.values,
                     rowLabels=stats_combined.index,
                     colLabels=stats_combined.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    # Customize table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Hide axes for the table subplot
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return stats_combined

def remove_outliers(df, column, n_std=1.5):
    """
    Remove outliers using the IQR method
    n_std: number of IQRs to use for the threshold (default is 1.5)
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - n_std * IQR
    upper_bound = Q3 + n_std * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Usage examples:
# With default settings (no max length, log scale)
#stats_df = plot_length_boxplots_with_limits_horizontal(df_multi)

# With max length of 1000 characters
#stats_df = plot_length_boxplots_with_limits_horizontal(df_multi, max_length=1000)

# With max length and linear scale
#stats_df = plot_length_boxplots_with_limits_horizontal(df_multi, max_length=1000, log_scale=False)


from enum import Enum

class FeatureCategory:
    MISSING_GROUND_TRUTH = "MISSING_GROUND_TRUTH"
    SHORT_TEXT = "SHORT_TEXT"
    LONG_TEXT = "LONG_TEXT"

def categorize_features(df, null_percentage_threshold=70, length_mean_threshold=50):
    feature_categories = {}
    
    for entity in df.entity.unique():
        # First check null_percentage
        null_percentage = df[
            (df.entity == entity) & 
            (df.metric_name == 'null_percentage')
        ]['metric_value'].values[0]
        
        if null_percentage > null_percentage_threshold:
            feature_categories[entity] = FeatureCategory.MISSING_GROUND_TRUTH
            continue
            
        # If not high null percentage, check length_mean
        length_mean = df[
            (df.entity == entity) & 
            (df.metric_name == 'length_mean')
        ]['metric_value'].values[0]
        
        if length_mean < length_mean_threshold:
            feature_categories[entity] = FeatureCategory.SHORT_TEXT
        else:
            feature_categories[entity] = FeatureCategory.LONG_TEXT
            
    return feature_categories

#feature_categories = categorize_features(results_df)
