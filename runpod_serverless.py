import runpod, os, yaml, dataset_downloader
from toolkit.job import get_job
from runpod.serverless.utils import rp_download, upload_file_to_bucket

S3_SCHEMA = {'accessId': {'type': str,'required': True},'accessSecret': {'type': str,'required': True},'bucketName': {'type': str,'required': True},'endpointUrl': {'type': str,'required': True}}

def edityaml(job_input):
    if job_input['gender'].lower() in ['woman','female','girl']:
        gender = 'woman'
        prompts = [f"{job_input['trigger_word']} stands in front of a mirror, capturing a selfie. The image quality is grainy, with a slight blur softening the details. The lighting is dim, casting shadows that obscure her features. The room is cluttered, with clothes strewn across the bed and an unmade blanket. her expression is casual, full of concentration, while the old iPhone struggles to focus, giving the photo an authentic, unpolished feel. The mirror shows smudges and fingerprints, adding to the raw, everyday atmosphere of the scene.",f"{job_input['trigger_word']} holding a coffee cup, in a beanie, sitting at a cafe",f"Mid-shot portrait of geraldine viswanathan in an hi tech office",f"Candid street photography, beautiful {job_input['trigger_word']}, night time, street smart, chic crop top, distressed jeans, wavy hair in light wind, natural lights, neon street lights, long exposure time, shallow depth of field, film grain, realism",f"Photorealistic photography of {job_input['trigger_word']}, close-up, mouth focus, extremely high detail, intricate textures, extremely dark surrounding, extremely low light, nighttime setting, ultra realistic shadows, ethereal, flowing hair illuminated by glowing butterflies, seductive smile, looking at viewer while playing with her hair, vibrant colors, dreamlike atmosphere, captivating expression, magical forest background, fiery glow, enchanted lighting, expressive eyes reflecting the glowing light, soft, ethereal glow around the figure, delicate and detailed facial features, golden light, intricate patterns in the background",f"{job_input['trigger_word']} playing the guitar, on stage, singing a song, laser lights, punk rocker"]
    elif job_input['gender'].lower() in ['man','male','boy']:
        gender = 'man'
        prompts = [f"{job_input['trigger_word']} stands in front of a mirror, capturing a selfie. The image quality is grainy, with a slight blur softening the details. The lighting is dim, casting shadows that obscure his features. The room is cluttered, with clothes strewn across the bed and an unmade blanket. his expression is casual, full of concentration, while the old iPhone struggles to focus, giving the photo an authentic, unpolished feel. The mirror shows smudges and fingerprints, adding to the raw, everyday atmosphere of the scene.",f"{job_input['trigger_word']} holding a coffee cup, in a beanie, sitting at a cafe",f"Mid-shot portrait of geraldine viswanathan in an hi tech office",f"Candid street photography, beautiful {job_input['trigger_word']}, night time, street smart, chic crop top, distressed jeans, wavy hair in light wind, natural lights, neon street lights, long exposure time, shallow depth of field, film grain, realism",f"Photorealistic photography of {job_input['trigger_word']}, close-up, mouth focus, extremely high detail, intricate textures, extremely dark surrounding, extremely low light, nighttime setting, ultra realistic shadows, ethereal, flowing hair illuminated by glowing butterflies, seductive smile, looking at viewer while playing with his hair, vibrant colors, dreamlike atmosphere, captivating expression, magical forest background, fiery glow, enchanted lighting, expressive eyes reflecting the glowing light, soft, ethereal glow around the figure, delicate and detailed facial features, golden light, intricate patterns in the background",f"{job_input['trigger_word']} playing the guitar, on stage, singing a song, laser lights, punk rocker"]
    file_path = f"config/{gender}.yaml"
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    data['config']['name'] = job_input['lora_file_name']
    data['config']['process'][0]['sample']['prompts'] = prompts
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
            if job_input['gender'].lower() in ['woman','female','girl']:
                job = get_job('config/woman.yaml', None)
            elif job_input['gender'].lower() in ['man','male','boy']:
                job = get_job('config/man.yaml', None)
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
