"""
Utility functions for model fine-tuning and management.

This module provides essential utilities for managing the fine-tuning process:
- Checkpoint handling: For training recovery and model selection
- AWS integration: For robust cloud resource management
"""

import os
import re
import json
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import boto3
import sagemaker

from pathlib import Path

def setup_directories(*args, **kwargs):
    """
    Create directories specified by positional and keyword arguments using pathlib.

    This function takes a list of directory paths as positional arguments and
    keyword arguments, and creates each directory (with parents) if they don't exist.

    Args:
        *args: Positional arguments representing directory paths
        **kwargs: Keyword arguments representing directory paths

    Example usage:
        setup_directories(
            'path/to/output_dir',
            'path/to/checkpoint_dir',
            additional_dir='path/to/additional_dir'
        )
    """
    all_paths = [Path(p) for p in args + tuple(kwargs.values())]
    
    for path in all_paths:
        path.mkdir(parents=True, exist_ok=True)
        
def check_checkpoints_directory(path: str) -> bool:
    """Verify checkpoint directory for training recovery exists
    
    Checkpoint verification is essential for:
    - Enabling training resumption after interruption
    - Preventing data loss
    - Ensuring training continuity
    
    Args:
        path: Path to checkpoints directory
        
    Returns:
        bool: True if directory exists and contains files
    """
    return os.path.exists(path) and os.path.isdir(path) and bool(os.listdir(path))

def parse_version_dir(dirname: str) -> Optional[Tuple[int, int, int, str]]:
    """Extract version information for model iteration tracking.
    
    Version parsing enables:
    - Chronological tracking of model iterations
    - Identification of training runs
    - Management of multiple training sessions
    
    Args:
        dirname: Directory name to parse
        
    Returns:
        Optional tuple of (version number, date, time, dirname)
    """
    pattern = re.compile(r"v(\d+)-(\d{8})-(\d{6})")
    match = pattern.match(dirname)
    if match:
        version = int(match.group(1))
        date = int(match.group(2))
        time = int(match.group(3))
        return (version, date, time, dirname)
    return None

def parse_checkpoint_dir(dirname: str) -> Optional[Tuple[int, str]]:
    """Extract checkpoint information for training progress tracking.
    
    Checkpoint parsing is necessary for:
    - Identifying training progress points
    - Managing model snapshots
    - Enabling selective checkpoint loading
    
    Args:
        dirname: Directory name to parse
        
    Returns:
        Optional tuple of (checkpoint number, dirname)
    """
    pattern = re.compile(r"checkpoint-(\d+)")
    match = pattern.match(dirname)
    if match:
        return (int(match.group(1)), dirname)
    return None

def find_latest_version_dir(model_dir: str) -> Optional[str]:
    """Locate most recent model version for training continuation.
    
    Finding the latest version is crucial for:
    - Ensuring training continues from most recent state
    - Maintaining version continuity
    - Preventing accidental use of outdated versions
    
    Args:
        model_dir: Path to model directory
        
    Returns:
        Optional string of latest version directory name
    """
    version_dirs = [
        parsed
        for dirname in os.listdir(model_dir)
        if (parsed := parse_version_dir(dirname)) is not None
    ]
    return max(version_dirs)[3] if version_dirs else None

def find_latest_checkpoint(version_dir: str) -> Optional[str]:
    """Locate most recent checkpoint for optimal training resumption.
    
    Latest checkpoint identification enables:
    - Efficient training recovery
    - Prevention of progress loss
    - Optimal model state restoration
    
    Args:
        version_dir: Path to version directory
        
    Returns:
        Optional string of latest checkpoint directory name
    """
    checkpoint_dirs = [
        parsed
        for dirname in os.listdir(version_dir)
        if (parsed := parse_checkpoint_dir(dirname)) is not None
    ]
    return max(checkpoint_dirs)[1] if checkpoint_dirs else None

def get_latest_sagemaker_training_job(job_name_prefix: str) -> Dict[str, Any]:
    """Retrieve most recent training information for monitoring and management.
    
    Latest job retrieval is essential for:
    - Training progress monitoring
    - Resource utilization tracking
    - Cost management
    - Error diagnosis
    
    Args:
        job_name_prefix: Prefix of training job name to search for
        
    Returns:
        Dictionary containing job description
        
    Raises:
        Exception: If no matching training job is found
    """
    sm_client = boto3.client('sagemaker')
    
    try:
        response = sm_client.list_training_jobs(
            NameContains=job_name_prefix,
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=1
        )
        
        if not response['TrainingJobSummaries']:
            raise Exception(f"No training jobs found with prefix: {job_name_prefix}")
            
        job_name = response['TrainingJobSummaries'][0]['TrainingJobName']
        return sm_client.describe_training_job(TrainingJobName=job_name)
        
    except Exception as e:
        raise Exception(f"Error getting latest training job: {str(e)}")

def get_s3_suffix(s3_url: str) -> str:
    """Extract filename from S3 URL for local file management.
        
    Args:
        s3_url: Full S3 URL
        
    Returns:
        Suffix path string
    """
    return s3_url.split('/')[-1]

def find_best_model_checkpoint(logging_file: str) -> Optional[str]:
    """Identify optimal model checkpoint based on evaluation metrics.
    
    Best checkpoint identification is crucial for:
    - Selecting the most performant model version
    - Optimizing model deployment
    
    Args:
        logging_file: Path to logging file
        
    Returns:
        Optional string path to best checkpoint
        
    Raises:
        Exception: If logging file cannot be read
    """
    try:
        best_loss = float('inf')
        best_checkpoint = None
        
        with open(logging_file, encoding="utf-8") as f:
            for line in f:
                try:
                    log_entry = json.loads(line)
                    if 'eval_loss' in log_entry:
                        if log_entry['eval_loss'] < best_loss:
                            best_loss = log_entry['eval_loss']
                            best_checkpoint = log_entry.get('checkpoint')
                except json.JSONDecodeError:
                    continue
                    
        return best_checkpoint
        
    except Exception as e:
        raise Exception(f"Error reading logging file: {str(e)}")

def find_latest_checkpoint_path(checkpoint_dir:str) -> str | None:
    """
    Finds the path to the latest checkpoint file in the specified checkpoint directory.

    This function checks if the specified checkpoint directory exists and contains
    valid checkpoint versions. If a valid directory and version are found, it retrieves
    the latest checkpoint file within the most recent version directory.

    Args:
        checkpoint_dir (str): The path to the checkpoint directory.

    Returns:
        str | None: The full path to the latest checkpoint file if found, otherwise None.

    Example:
        >>> find_latest_checkpoint_path('/path/to/checkpoints')
        '/path/to/checkpoints/v0-20250213-052238/checkpoint-42'
    """
    if check_checkpoints_directory(checkpoint_dir):
        model_dir = checkpoint_dir
        if os.path.exists(model_dir):
            latest_version = find_latest_version_dir(model_dir)
            if latest_version:
                latest_dir = os.path.join(model_dir, latest_version)
                latest_checkpoint = find_latest_checkpoint(latest_dir)
                if latest_checkpoint:
                    full_checkpoint_path = os.path.join(latest_dir, latest_checkpoint)
                    return full_checkpoint_path
    return None