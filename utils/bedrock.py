import pandas as pd
import matplotlib.pyplot as plt
from .helpers import find_file_with_resource


def get_fine_tunable_vision_models(bedrock_client):
    fine_tunable_models = bedrock_client.list_foundation_models(
        byOutputModality="TEXT",
        byCustomizationType="FINE_TUNING"
    )["modelSummaries"]
    
    fine_tunable_models = [model for model in fine_tunable_models if 'IMAGE' in model.get("inputModalities", [])]

    return fine_tunable_models

def customization_job_status(job_arn, bedrock):
    try:
        response = bedrock.get_model_customization_job(jobIdentifier=job_arn)

        status = response.get('status', 'Failed')
        print(f"Customization job status: {status}")
        if status == 'Completed':
            return 'Success'
        elif status == 'Failed':
            return 'Failed'
        else:
            return 'InProgress'
    
    except Exception as e:
        print(f"Error checking customization job status: {e}")
        return 'Failed'


def deployment_status(deployment_arn, bedrock):
    """
    Check the status of a custom model deployment
    
    Parameters:
    -----------
    deployment_arn : str
        ARN of the deployment to check
    bedrock: boto3 bedrock client
        
    Returns:
    --------
    status : str
        Current status of the deployment
    """
    try:
        response = bedrock.get_custom_model_deployment(
            customModelDeploymentIdentifier=deployment_arn
        )
        status = response.get('status')
        print(f"Deployment status: {status}")
        if status == 'Active':
            return 'Success'
        elif status == 'Failed':
            return 'Failed'
        else:
            return 'InProgress'
    
    except Exception as e:
        print(f"Error checking deployment status: {e}")
        return 'Failed'
    
# Function to wait for deployment to be ready
def wait_for(check_fn, max_wait_seconds=1800, check_interval=60, **kwargs):
    """
    Wait for an asynchronous API to complete
    
    Parameters:
    -----------
    check_fn : Callable
        Function to call every interval to check if the async operation completed. Needs to return one of the following strings: "Success", "Failed", "InProgress"
    max_wait_seconds : int
        Maximum wait time in seconds (default: 1800s = 30 minutes)
    check_interval : int
        Interval between checks in seconds (default: 60s)
    **kwargs: arguments to pass to check_fn
        
    Returns:
    --------
    success : bool
        True if the async job completed, False otherwise
    """
    import time
    from tqdm.notebook import tqdm
    
    start_time = time.time()
    end_time = start_time + max_wait_seconds
    
    print(f"Waiting for {check_fn.__name__} to complete (max wait time: {max_wait_seconds/60:.1f} minutes)")
    
    # Create a progress bar for the wait time
    with tqdm(total=max_wait_seconds, desc=f"Waiting for {check_fn.__name__}", unit="sec") as pbar:
        elapsed = 0
        while time.time() < end_time:
            status = check_fn(**kwargs)
            # status = check_deployment_status(deployment_arn)
            
            if status == "Success":
                print(f"\n✅ {check_fn.__name__} succeeded {(time.time() - start_time)/60:.1f} minutes")
                return True
                
            if status == "Failed":
                print(f"\n❌ {check_fn.__name__} failed or was deleted.")
                return False
                
            # Update progress bar with time elapsed since last check
            new_elapsed = int(time.time() - start_time)
            pbar.update(new_elapsed - elapsed)
            elapsed = new_elapsed
            
            # Wait before checking again
            time.sleep(check_interval)
    
    print(f"\n⚠️ Timed out after waiting {max_wait_seconds/60:.1f} minutes. {check_fn.__name__} may still be in progress.")
    return False


def plot_loss(df, loss_column, output_file=None):
    ''' This function plots training loss using the default model output file 'step_wise_training_metrics.csv' generated from the finetuning job'''
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['step_number'], df[loss_column], 'b-', linewidth=2)
    
    # Customize the plot
    plt.title(f'{loss_column} vs step number', fontsize=14)
    plt.xlabel('step number', fontsize=12)
    plt.ylabel(loss_column, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add some padding to the axes
    plt.margins(x=0.02)
    
    # Save or display the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved as {output_file}")
    else:
        plt.show()


def load_and_plot_metric(output_s3_prefix, metric_csv_name, metric_column, output_file=None):
    metric_s3 = find_file_with_resource(output_s3_prefix, metric_csv_name)
    if metric_s3 and len(metric_s3) > 0:
        metric_s3 = metric_s3[0]
        metric_df = pd.read_csv(metric_s3)
        plot_loss(metric_df, metric_column, output_file)
