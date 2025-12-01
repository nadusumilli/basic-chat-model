from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

# Set up credentials for WatsonxLLM.
credentials = Credentials(
    # "project_id": "51d55348-c06b-4b08-8ec2-1e17245c7729",
    # France location url for watson.
    url="https://eu-de.ml.cloud.ibm.com",
    # uncomment the field below and replace the api_key with your actual Watsonx API key
    api_key="JHobM3SL1UkmHD0_m91Af5W3mOiizNocL2fjJSQVlFl8"
)

params = {
    GenTextParamsMetaNames.DECODING_METHOD: "greedy",
    GenTextParamsMetaNames.MAX_NEW_TOKENS: 100
}

model = ModelInference(
    model_id='ibm/granite-4-h-small',
    params=params,
    credentials=credentials,
    project_id="51d55348-c06b-4b08-8ec2-1e17245c7729"
)

text = """
Only reply with the answer. What is the capital of Canada?
"""

print(model.generate(text)['results'][0]['generated_text'])
