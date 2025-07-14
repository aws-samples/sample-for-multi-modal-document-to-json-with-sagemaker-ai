import sys
from dataclasses import dataclass
from packaging.version import Version
import requests
import re
from bs4 import BeautifulSoup
from typing import Optional
import subprocess
import os
import json
import time

@dataclass
class SageMakerDistribution:
    image_version: str
    sagemaker_python_sdk: str


def get_python_version(major, minor, *args,**kwargs) -> Version:
    return Version(f"{major}.{minor}")


# list of supported images: https://github.com/aws/sagemaker-distribution/blob/main/support_policy.md#supported-image-versions
MAP_PY_VER_TO_SM_DISTRIBUTION_VER = {
   get_python_version(3, 11):SageMakerDistribution("2.6.0","2.243.2"),   # https://github.com/aws/sagemaker-distribution/blob/main/build_artifacts/v2/v2.6/v2.6.0/CHANGELOG-gpu.md
    get_python_version(3, 12):SageMakerDistribution("3.1.0","2.244.2")  # https://github.com/aws/sagemaker-distribution/blob/main/build_artifacts/v3/v3.1/v3.1.0/CHANGELOG-gpu.md
}

FALLBACK_VERSION = get_python_version(3,12)


def get_sagemaker_distribution(py_version) -> SageMakerDistribution:
    
    sm_distro_version = MAP_PY_VER_TO_SM_DISTRIBUTION_VER.get(get_python_version(*py_version), None)
    if not sm_distro_version:
        sm_distro_version = MAP_PY_VER_TO_SM_DISTRIBUTION_VER[FALLBACK_VERSION]
        print(
            f"""[Warning] No matching SageMaker distribution found for python version {str(sys.version_info)}. Your local Python version needs to match the Python version in the training image. 
    
    Falling back to SageMaker distribution v{sm_distro_version.image_version} for training image. 
    This might lead to errors. If you see Python version mismatch errors during training please upgrade or downgrade your local Python version to a tested Python version: {[str(key) for key in MAP_PY_VER_TO_SM_DISTRIBUTION_VER.keys()]}.
    """
        )
    return sm_distro_version


def get_aws_account_id_for_region(region: str) -> Optional[str]:
    """
    Fetch the AWS Account ID for SageMaker Distribution Images for a given region.
    
    Args:
        region (str): AWS region (e.g., 'us-east-1', 'eu-west-1')
    
    Returns:
        str: AWS Account ID (12-digit string) or None if region not found
    """
    url = "https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks-available-images.html"

    if region == "us-east-1":
        return "885854791233"
    elif region == "us-west-2":
        return "542918446943"
    
    
    try:
        # Fetch the webpage
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the table containing the region mappings
        table = soup.find('table')
        if not table:
            return None
        
        # Parse table rows
        rows = table.find_all('tr')
        
        for row in rows[1:]:  # Skip header row
            cells = row.find_all('td')
            if len(cells) >= 3:
                table_region = cells[0].get_text(strip=True)
                sagemaker_distribution_arn = cells[2].get_text(strip=True)
                
                # Check if this is the region we're looking for
                if table_region == region:
                    # Extract account ID from the SageMaker Distribution Image ARN Format
                    # Pattern: arn:aws:sagemaker:region:ACCOUNT_ID:image/resource-identifier
                    match = re.search(r':(\d{12}):', sagemaker_distribution_arn)
                    if match:
                        return match.group(1)
        
        return None
        
    except requests.RequestException as e:
        print(f"Error fetching webpage: {e}")
        return None
    except Exception as e:
        print(f"Error parsing content: {e}")
        return None



def is_docker_installed():
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def is_docker_compose_installed():
    try:
        subprocess.run(["docker-compose", "--version"], check=True, capture_output=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def load_json_file(file_path):
    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        print(f"File {file_path} is not a valid file")
        return None

SAGEMAKER_STUDIO_METADATA = "/opt/ml/metadata/resource-metadata.json"
DOCKER_ENABLED = "ENABLED"
DOCKER_DISABLED = "DISABLED"

def check_and_enable_docker_access_sagemaker_studio(use_local_mode, session):
    if use_local_mode:
        resource_metadata = load_json_file(SAGEMAKER_STUDIO_METADATA)
        if resource_metadata:
            docker_access_disabled = True
            docker_access = None
            try:
                domain_id = resource_metadata["DomainId"]
                sm_client = session.boto_session.client("sagemaker")
                domain = sm_client.describe_domain(DomainId=domain_id)
                if docker_access_disabled := ((docker_access := domain["DomainSettings"]["DockerSettings"]["EnableDockerAccess"]) == DOCKER_DISABLED):
                    print("Docker disabled on SageMaker Studio domain. Trying to enable docker access...")
                    sm_client.update_domain(
                        DomainId=domain_id,
                        DomainSettingsForUpdate={
                            'DockerSettings': {
                                'EnableDockerAccess': DOCKER_ENABLED
                            }
                        }
                    )
                    time.sleep(4)
                    domain = sm_client.describe_domain(DomainId=domain_id)
                    docker_access = domain["DomainSettings"]["DockerSettings"]["EnableDockerAccess"]
                    docker_access_disabled = (docker_access == DOCKER_DISABLED)
            except Exception as e: 
                print(e)
                
            print(f"SageMaker Studio domain ({domain_id}) docker access: {docker_access}")
            if docker_access_disabled:
                print("Failed to enable Docker Access on SageMaker Studio domain. Please enable it manually or ask your administrator. Docker access is required to run in local mode. https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-local-get-started.html#studio-updated-local-enable")