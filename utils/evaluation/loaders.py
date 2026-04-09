"""Data loading with intelligent format detection."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, List
import pandas as pd
import json
import os
import tarfile
import tempfile
import shutil
from pathlib import Path
import regex
import boto3

# JSON pattern for extracting JSON from responses
regex.DEFAULT_VERSION = regex.VERSION1
JSON_PATTERN = regex.compile(r'\{(?:[^{}"]|"(?:\\.|[^"\\])*"|(?R))*\}')


def parse_json_response(input_string: str) -> Dict:
    """Extract and parse JSON from model response."""
    if not input_string:
        return {}
    
    json_match = JSON_PATTERN.search(input_string)
    if json_match:
        try:
            parsed_json = json.loads(json_match.group())
            # Handle unicode escapes
            for k, v in parsed_json.items():
                if isinstance(v, str):
                    try:
                        parsed_json[k] = bytes(v, "utf-8").decode("unicode_escape")
                    except:
                        pass
            return parsed_json
        except json.JSONDecodeError:
            pass
    return {}


def to_dict(text: str) -> Dict:
    """Convert text to dict, handling nested JSON strings."""
    try:
        d = json.loads(text)
        if isinstance(d, str):
            try:
                d = json.loads(d)
            except:
                return {}
        return d if isinstance(d, dict) else {}
    except:
        return {}


class ResultLoader(ABC):
    """Base class for format-specific loaders."""
    
    @abstractmethod
    def load(self, location: str) -> List[Dict]:
        """Load JSONL content from location."""
        pass


class TarArchiveLoader(ResultLoader):
    """Handle SageMaker tar.gz files."""
    
    def load(self, location: str) -> List[Dict]:
        """Extract tar and load JSONL files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract tar
            with tarfile.open(location, 'r:gz') as tar:
                tar.extractall(tmpdir)
            
            # Find JSONL files
            jsonl_files = list(Path(tmpdir).rglob('*.jsonl'))
            if not jsonl_files:
                raise ValueError(f"No JSONL files found in tar archive: {location}")
            
            # Load all JSONL files
            data = []
            for jsonl_file in jsonl_files:
                data.extend(_read_jsonl(str(jsonl_file)))
            return data


class JsonlFileLoader(ResultLoader):
    """Handle direct JSONL files."""
    
    def load(self, location: str) -> List[Dict]:
        """Load JSONL file."""
        return _read_jsonl(location)


class DirectoryLoader(ResultLoader):
    """Handle directories with JSONL files."""
    
    def load(self, location: str) -> List[Dict]:
        """Load all JSONL files from directory."""
        jsonl_files = list(Path(location).rglob('*.jsonl'))
        if not jsonl_files:
            raise ValueError(f"No JSONL files found in directory: {location}")
        
        data = []
        for jsonl_file in jsonl_files:
            data.extend(_read_jsonl(str(jsonl_file)))
        return data


def _read_jsonl(file_path: str) -> List[Dict]:
    """Read JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


class ResponseParser(ABC):
    """Base class for conversation format parsers."""
    
    @abstractmethod
    def parse(self, data: Dict, dataset_base_dir: str = "") -> Dict:
        """Parse conversation format to standardized structure."""
        pass


class SwiftResponseParser(ResponseParser):
    """Parse SWIFT conversation format."""
    
    def parse(self, data: Dict, dataset_base_dir: str = "") -> Dict:
        """Extract response, labels, and images from SWIFT format."""
        result = {
            'response_raw': data.get('response', ''),
            'response_parsed': parse_json_response(data.get('response', '')),
            'labels': to_dict(data.get('labels', '{}')),
            'images': data.get('images', []),
            'messages': data.get('messages', [])
        }
        return result


class NovaResponseParser(ResponseParser):
    """Parse Amazon Nova conversation format."""
    
    def parse(self, data: Dict, dataset_base_dir: str = "") -> Dict:
        """Extract response, labels, and images from Nova format."""
        result = {
            'response_raw': data.get('response', ''),
            'response_parsed': parse_json_response(data.get('response', '')),
            'labels': to_dict(data.get('labels', '{}')),
            'images': self._extract_images_from_messages(data.get('messages', []), dataset_base_dir),
            'messages': data.get('messages', [])
        }
        return result
    
    def _extract_images_from_messages(self, messages: List[Dict], dataset_base_dir: str) -> List[Dict]:
        """Extract image paths from Nova message format."""
        if not messages:
            return []
        
        images = []
        for msg in messages:
            if "content" in msg:
                for content in msg["content"]:
                    if "image" in content:
                        img = content["image"]
                        s3_location = img.get("source", {}).get("s3Location", {}).get("uri", None)
                        if s3_location:
                            # Remove s3:// prefix and construct local path
                            local_path = s3_location.replace("s3://", "")
                            if dataset_base_dir:
                                local_path = os.path.join(dataset_base_dir, local_path)
                            images.append({"path": local_path})
        return images


class GenericResponseParser(ResponseParser):
    """Fallback parser for unknown formats."""
    
    def parse(self, data: Dict, dataset_base_dir: str = "") -> Dict:
        """Generic parsing."""
        result = {
            'response_raw': data.get('response', ''),
            'response_parsed': parse_json_response(data.get('response', '')),
            'labels': to_dict(data.get('labels', '{}')),
            'images': data.get('images', []),
            'messages': data.get('messages', [])
        }
        return result


def detect_format(location: str) -> str:
    """Identify tar/jsonl/directory from path."""
    if location.startswith('s3://'):
        # For S3, check extension
        if location.endswith('.tar.gz') or location.endswith('.tgz'):
            return 'tar'
        elif location.endswith('.jsonl'):
            return 'jsonl'
        else:
            return 'tar'  # Default for S3
    
    if os.path.isfile(location):
        if location.endswith('.tar.gz') or location.endswith('.tgz'):
            return 'tar'
        elif location.endswith('.jsonl'):
            return 'jsonl'
    elif os.path.isdir(location):
        return 'dir'
    
    raise ValueError(f"Cannot detect format for location: {location}")


def detect_parser(data: List[Dict]) -> str:
    """Identify conversation format from JSONL structure."""
    if not data:
        return 'generic'
    
    sample = data[0]
    messages = sample.get('messages', [])
    
    if messages and isinstance(messages, list) and len(messages) > 0:
        first_msg = messages[0]
        # Check for Nova format (has 'content' with nested structure)
        if 'content' in first_msg and isinstance(first_msg['content'], list):
            for content in first_msg['content']:
                if 'image' in content and 'source' in content['image']:
                    return 'nova'
    
    # Check for SWIFT format
    if 'images' in sample:
        return 'swift'
    
    return 'generic'


def detect_model_name(location: str, data: List[Dict]) -> Optional[str]:
    """Extract model name from path/metadata."""
    # Try to extract from path
    path_lower = location.lower()
    
    # Common model patterns
    if 'qwen' in path_lower:
        if '2.5' in path_lower or '2-5' in path_lower:
            return 'Qwen/Qwen2.5-VL-3B-Instruct'
        return 'Qwen/Qwen2-VL-Instruct'
    elif 'llama' in path_lower:
        if '3.2' in path_lower or '3-2' in path_lower:
            return 'meta-llama/Llama-3.2-11B-Vision-Instruct'
        return 'meta-llama/Llama-3-Vision'
    elif 'nova' in path_lower:
        if 'lite' in path_lower:
            return 'amazon.nova-lite-v1:0'
        return 'amazon.nova-v1:0'
    
    # Try to extract from data
    if data and 'model_name' in data[0]:
        return data[0]['model_name']
    
    return None


def _download_from_s3(s3_uri: str, local_path: str):
    """Download file from S3."""
    s3 = boto3.client('s3')
    # Parse S3 URI
    parts = s3_uri.replace('s3://', '').split('/', 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ''
    
    s3.download_file(bucket, key, local_path)


def process_results(source, dataset_base_dir: str = "") -> pd.DataFrame:
    """Auto-detect and process single source."""
    from .config import ResultSource
    
    location = source.location
    
    # Handle S3 URIs
    if location.startswith('s3://'):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tar.gz') as tmp:
            _download_from_s3(location, tmp.name)
            location = tmp.name
    
    # Detect format and load
    fmt = detect_format(location)
    if fmt == 'tar':
        loader = TarArchiveLoader()
    elif fmt == 'jsonl':
        loader = JsonlFileLoader()
    else:  # dir
        loader = DirectoryLoader()
    
    data = loader.load(location)
    
    # Detect parser
    parser_type = detect_parser(data)
    if parser_type == 'swift':
        parser = SwiftResponseParser()
    elif parser_type == 'nova':
        parser = NovaResponseParser()
    else:
        parser = GenericResponseParser()
    
    # Parse all records
    parsed_data = []
    for idx, record in enumerate(data):
        parsed = parser.parse(record, dataset_base_dir)
        parsed['file_id'] = idx
        parsed['name'] = source.name
        parsed['model_name'] = source.model_name or detect_model_name(source.location, data)
        parsed_data.append(parsed)
    
    # Clean up temp file if S3
    if source.location.startswith('s3://') and os.path.exists(location):
        os.unlink(location)
    
    return pd.DataFrame(parsed_data)


class ResultsRegistry:
    """Manages multiple inference results."""
    
    def __init__(self):
        self.sources = []
    
    def add_from_csv(self, path: str, dataset_base_dir: str = "", column_mapping: dict = None):
        """Load sources from CSV file.
        
        Args:
            path: Path to CSV file
            dataset_base_dir: Base directory for resolving image paths
            column_mapping: Optional dict mapping expected keys to CSV column names.
                          Default: {'name': 'human_name', 'location': 'inference_results_s3', 'model_name': 'model'}
        """
        from .config import ResultSource
        
        if column_mapping is None:
            column_mapping = {'name': 'human_name', 'location': 'inference_results_s3', 'model_name': 'model'}
        
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            source = ResultSource(
                name=row.get(column_mapping.get('name', 'name'), row.get('name')),
                location=row.get(column_mapping.get('location', 'location'), row.get('location')),
                model_name=row.get(column_mapping.get('model_name', 'model_name'), row.get('model_name'))
            )
            self.sources.append((source, dataset_base_dir))
    
    def add_source(self, source, dataset_base_dir: str = ""):
        """Add individual ResultSource."""
        self.sources.append((source, dataset_base_dir))
    
    def load_all(self) -> pd.DataFrame:
        """Process all sources, return unified DataFrame."""
        all_data = []
        for source, dataset_base_dir in self.sources:
            df = process_results(source, dataset_base_dir)
            all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        return pd.concat(all_data, ignore_index=True)
