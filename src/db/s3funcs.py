import os
import boto3

s3 = boto3.client('s3')

bucket_name = 'candidate-proctoring'
download_dir = 'attendee_db/'  # Update this to your desired download location

valid_extensions = ['.mp4', 'webm', '.jpg', '.jpeg', '.png']

def list_candidates(bucket):
    """List all candidates in the S3 bucket."""
    response = s3.list_objects_v2(Bucket=bucket, Delimiter='/')
    candidates = [prefix['Prefix'].strip('/') for prefix in response.get('CommonPrefixes', [])]
    good_candidates = []
    for candidate in candidates:
        if not check_proctoring_alerts(bucket_name, candidate):
            good_candidates.append(candidate)
    return good_candidates

def check_proctoring_alerts(bucket, candidate):
    """Check if 'proctoring_alerts.log' exists in the candidate's logs folder."""
    log_file = f'{candidate}/log/proctoring_alerts.log'
    try:
        s3.head_object(Bucket=bucket, Key=log_file)
        return True
    except:
        return False

def retrieve_file_paths(bucket, candidate):
    """Retrieve image and video files for the candidate."""
    video_prefix = f'{candidate}/video/'
    image_prefix = f'{candidate}/image/'
    
    video_files = []
    image_files = []

    # List video files
    video_response = s3.list_objects_v2(Bucket=bucket, Prefix=video_prefix)
    for obj in video_response.get('Contents', []):
        if any(obj['Key'].endswith(ext) for ext in valid_extensions):
            video_files.append(obj['Key'])

    # List image files
    image_response = s3.list_objects_v2(Bucket=bucket, Prefix=image_prefix)
    for obj in image_response.get('Contents', []):
        if any(obj['Key'].endswith(ext) for ext in valid_extensions):
            image_files.append(obj['Key'])

    return video_files, image_files

def download_s3_files(bucket, file_path):
    """Download files from S3 to the local directory."""
    if any(file_path.split('/')[-1].endswith(ext) for ext in valid_extensions):
        local_path = os.path.join(download_dir, file_path.split('/')[-1])
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket, file_path, local_path)
        print(f"Downloaded {file_path.split('/')[-1]} to {local_path}")

def upload_file_to_s3(file_path, bucket_name, candidate, object_name=None):
    if object_name is None:
        object_name = f"{candidate}/log/{file_path.split('/')[-1]}"
    try:
        s3.upload_file(file_path, bucket_name, object_name)
        print(f"File {file_path.split('/')[-1]} uploaded to {bucket_name}/{object_name}")
        return True
    except Exception as e:
        print(f"Failed to upload {file_path.split('/')[-1]} to {bucket_name}/{object_name}. Error: {e}")
        return False