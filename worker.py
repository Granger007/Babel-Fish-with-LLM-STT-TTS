import requests
import os
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from requests.auth import HTTPBasicAuth

# Set your Watsonx Project ID and credentials
PROJECT_ID = "skills-network"
WATSON_TTS_APIKEY = os.environ.get("WATSON_TTS_APIKEY")  # Set this in your shell
WATSON_TTS_URL = "https://api.us-south.text-to-speech.watson.cloud.ibm.com"  # Adjust if needed

# Credentials for Watsonx LLM
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
    # "apikey": YOUR_WATSONX_APIKEY if needed
}

# Choose your model
model_id = ModelTypes.FLAN_UL2

parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 1024
}

# Define the LLM
model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=PROJECT_ID
)

def speech_to_text(audio_binary):
    base_url = 'https://sn-watson-stt.labs.skills.network'
    api_url = base_url + '/speech-to-text/api/v1/recognize'
    params = {
        'model': 'en-US_Multimedia',
    }

    response = requests.post(api_url, params=params, data=audio_binary).json()

    text = 'null'
    while bool(response.get('results')):
        print('Speech-to-Text response:', response)
        text = response.get('results').pop().get('alternatives').pop().get('transcript')
        print('recognised text: ', text)
        return text

def text_to_speech(text, voice="en-US_LisaV3Voice"):
    api_url = f"{WATSON_TTS_URL}/v1/synthesize"

    headers = {
        'Accept': 'audio/wav',
        'Content-Type': 'application/json'
    }

    params = {
        'voice': voice
    }

    payload = {
        'text': text
    }

    response = requests.post(
        api_url,
        headers=headers,
        json=payload,
        params=params,
        auth=HTTPBasicAuth('apikey', WATSON_TTS_APIKEY)
    )

    if response.status_code != 200:
        print("TTS error:", response.status_code, response.text)
        return b''

    return response.content

def watsonx_process_message(user_message):
    prompt = f"""You are an assistant helping translate sentences from English into Spanish.
    Translate the query to Spanish: ```{user_message}```."""
    
    response_text = model.generate_text(prompt=prompt)
    print("watsonx response:", response_text)
    return response_text
