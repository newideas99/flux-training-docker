import runpod, os, yaml, dataset_downloader
from toolkit.job import get_job
from runpod.serverless.utils import rp_download, upload_file_to_bucket

S3_SCHEMA = {'accessId': {'type': str,'required': True},'accessSecret': {'type': str,'required': True},'bucketName': {'type': str,'required': True},'endpointUrl': {'type': str,'required': True}}

def edityaml(job_input):
    file_path = f"config/training-config-dev.yaml"
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    data['config']['name'] = job_input['lora_file_name']
    data['config']['process'][0]['trigger_word'] = job_input['trigger_word']
    data['config']['process'][0]['datasets'][0]['folder_path'] = job_input['dataset']
    with open(file_path, 'w') as file:
        yaml.safe_dump(data, file, default_flow_style=False)
    return True

def download(job_input):
    downloaded_input = dataset_downloader.file(job_input['data_url'])
    dataset = os.path.join(downloaded_input['extracted_path'])
    job_input['dataset'] = dataset
    return job_input

def train_lora(job):

    if 's3Config' in job:
        s3_config = job["s3Config"]
        job_input = job["input"]
        job_input = download(job_input)
        if edityaml(job_input) == True:
            job = get_job('config/training-config-dev.yaml', None)
            job.run()
            job.cleanup()
            lora_path = os.path.join('output', job_input['lora_file_name'])
            lora_paths = os.listdir(lora_path)
            job_output = {'lora_url': []}
            for file in lora_paths:
                if file.endswith('.safetensors'):
                    try:
                        lora_url = upload_file_to_bucket(file_name=file,file_location=os.path.join(lora_path, file),bucket_creds=s3_config,bucket_name=s3_config['bucketName'])
                        job_output['lora_url'].append(lora_url)
                    except Exception as e:
                        print(e)
                        return {'error':str(e)}
            return job_output
        else:
            return {'error':'Training YAML error'}
    else:
        return {"error": 'S3 config not set!!!!'}

runpod.serverless.start({"handler": train_lora})
