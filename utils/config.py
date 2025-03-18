"""
Configuration settings for model fine-tuning.

This module uses dataclasses to enforce type safety and immutability in configurations.
"""

from dataclasses import dataclass
from dataclasses_json import dataclass_json

from typing import List, Union
from .helpers import shorten_for_sagemaker_training_job

@dataclass
class ModelConfig:
    """Model configuration settings.
    
    Separate model configuration enables:
    - Easy model switching for experimentation
    - Clear model versioning
    - Simplified model comparison
    """
    model_type: str
    model_id: str
    
    @property
    def model_id_simple(self) -> str:
        """Generate simplified model identifier for file naming.
        
        Simplified IDs are crucial for:
        - Avoiding path length limitations
        - Maintaining consistent naming conventions
        - Preventing special character issues in filenames
        """
        return self.model_id.split("/")[-1].replace(".", "-").lower()

    def training_job_prefix(self, dataset_s3_prefix: str) -> str:
        """Generate prefix name for training job.
        """
        prefix = f"finetune-{self.model_id_simple}-{dataset_s3_prefix.replace('/','-')}"
        return shorten_for_sagemaker_training_job(prefix)
