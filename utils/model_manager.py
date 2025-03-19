"""
Model Management and Inference Module

This module handles model operations for machine learning inference at scale. This module helps with:
1. Model management
2. Checkpoint management is critical for using the best performing model version
3. Downloading and extracting of model artifacts

The module centralizes these concerns to ensure reliable model inference.
"""

from pathlib import Path
from typing import Union, Dict, Optional
import subprocess
from .helpers import (
    get_s3_suffix,
    find_latest_version_directory,
    find_best_model_checkpoint,
    merge_paths
)
import json
import pandas as pd
import shlex

class ModelManager:
    """Handles model-related operations including downloading and extracting model artifacts."""
    
    def __init__(self, model_weights_dir: str = "./models"):
        """
        Initialize with a model weights directory.
        
        We use a dedicated directory for model weights to:
        1. Keep model artifacts separate from application code
        2. Enable easy cleanup of downloaded models
        """
        self.model_weights_dir = Path(model_weights_dir)
        self.model_weights_dir.mkdir(exist_ok=True)
        
    def download_and_extract_model(self, model_s3_uri: str) -> Path:
        """
        Downloads model from S3 and extracts it.
        
        Args:
            model_s3_uri: S3 URI of the model artifacts
            
        Returns:
            Path to the extracted model directory
        """
        try:
            model_suffix_s3 = self._get_s3_suffix(model_s3_uri)
            model_destination = self.model_weights_dir / model_suffix_s3
            model_dest_dir = model_destination.parent
            
            self._download_from_s3(model_s3_uri, str(model_destination))
            
            self._extract_tar(str(model_destination), str(model_dest_dir))
            
            return model_dest_dir
        except Exception as e:
            print(f"Failed to download and extract model: {str(e)}")
            raise

    def download_from_hf_hub(self, model_id: str) -> Path:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_id, local_dir=str(self.model_weights_dir))
        return self.model_weights_dir
    
    @staticmethod
    def update_generation_config(directory: Path, guided_decoding: Optional[Dict]) -> None:
        """Update generation config with guided decoding settings"""
        if not guided_decoding:
            return
        
        # Ensure parent directory exists
        directory.mkdir(parents=True, exist_ok=True)
        
        config_path = directory / 'generation_config.json'
        
        # Create empty config if missing
        if not config_path.exists():
            config_path.write_text('{}')
        
        # Load and update config
        with config_path.open('r') as f:
            config = json.load(f)
        
        config.setdefault('guided_decoding', {}).update(guided_decoding)
        
        # Write updated config
        with config_path.open('w') as f:
            json.dump(config, f, indent=2)


    @staticmethod
    def construct_guided_decoding_config(dataset_dir: Path, guided_decoding: Optional[Union[Dict, str]]) -> Optional[Dict]:
        generation_config = None
        if guided_decoding:
            generation_config = guided_decoding
            if isinstance(guided_decoding, str):
                guided_decoding = dataset_dir / guided_decoding
                with guided_decoding.open('r') as f:
                    json_schema = json.load(f)
                    generation_config = {"json":json_schema}
        return generation_config


    @staticmethod
    def find_best_model_checkpoint(model_dir: Union[str, Path]) -> str:
        """
        Locates the best performing model checkpoint using pathlib for cross-platform safety,
        replaces specific prefixes in the checkpoint path, and merges paths to reconstruct
        the full checkpoint directory.
        
        Args:
            model_dir: Directory where model versions are stored.
        
        Returns:
            Full path to the best model checkpoint.
        """
        # Convert inputs to Path objects if necessary
        model_dir_path = Path(model_dir) if isinstance(model_dir, str) else model_dir
        # base_dir_path = Path(base_dir) if isinstance(base_dir, str) else base_dir
    
        # Find latest version directory using pathlib
        latest_version = find_latest_version_directory(model_dir_path)
        logging_file = model_dir_path / Path(latest_version) / "logging.jsonl"
    
        if not logging_file.exists():
            raise FileNotFoundError(f"Training logs not found at {logging_file}")
    
        # Find the best checkpoint using logging data
        best_model_checkpoint = find_best_model_checkpoint(logging_file)
        
        if not best_model_checkpoint:
            raise ValueError(
                f"Best model checkpoint not found. Please search the logs {logging_file} manually "
                "to find the path that stores the best model checkpoint."
            )
    
        # Replace "/opt/ml/model/" and "/opt/ml/checkpoints/" in the checkpoint path
        normalized_checkpoint = (
            best_model_checkpoint
            .replace("/opt/ml/model/", "")
            .replace("/opt/ml/checkpoints/", "")
        )
    
        # Merge paths to reconstruct the full checkpoint directory using updated merge_paths
        ckpt_dir = merge_paths(model_dir_path, normalized_checkpoint)
        
        # Validate and return the full path as a string
        print(f"Best model checkpoint located: {ckpt_dir}")
        return str(ckpt_dir)

   
            
    @staticmethod
    def _get_s3_suffix(s3_uri: str) -> str:
        """Extract the suffix from an S3 URI."""
        return get_s3_suffix(s3_uri)

    
    @staticmethod
    def _download_from_s3(source: str, destination: str) -> None:
        """Download a file from S3."""
        subprocess.run(["aws", "s3", "cp", source, destination, "--quiet"], check=True, shell=False)
        
    @staticmethod
    def _extract_tar(source: str, destination: str) -> None:
        """Extract a tar file."""
        subprocess.run(
            [
                "tar",
                "--warning=no-unknown-keyword",
                "-xzvf",
                source,
                "--directory",
                destination,
            ],
            stdout=subprocess.DEVNULL,
            check=True,
            shell=False
        )



def list_available_models(bucket_name: str, model_prefix: str) -> pd.DataFrame: 

    


    # Escape the inputs to prevent command injection
    escaped_model_prefix = shlex.quote(model_prefix)
    escaped_bucket_name = shlex.quote(bucket_name)

    # Remove the quotes added by shlex.quote() since this is going inside another string
    # and not directly as a command argument
    escaped_model_prefix = escaped_model_prefix.strip("'")
    escaped_bucket_name = escaped_bucket_name.strip("'")

    cmd = [
        'aws', 's3api', 'list-objects-v2',
        '--bucket', escaped_bucket_name,
        '--query', f"reverse(Contents[?starts_with(Key, '{escaped_model_prefix}') && contains(Key, 'model.tar.gz')]|sort_by(@, &LastModified))[].[LastModified,Size,Key]"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True, shell=False)
    data = json.loads(result.stdout)
    
    df = pd.DataFrame(data, columns=['LastModified', 'Size', 'Key'])
    df['Size'] = df['Size'].apply(lambda x: x/1024/1024)
    df['LastModified'] = pd.to_datetime(df['LastModified'])
    df = df[df['Size']>100] # finished adapter should be at least 100mb
    # increase width of column key for df,
    pd.set_option('display.max_colwidth', 200)
    return df
