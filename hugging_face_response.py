from huggingface_hub import InferenceClient


def gen_response(disease, hf_token):

    prompt = f"""
    You are an expert agricultural assistant.

    A plant disease classifier predicted:

    Disease: {disease}

    Provide:

    1. Disease overview
    2. Symptoms
    3. Causes
    4. Treatment
    5. Prevention tips

    Use simple language suitable for farmers.
    """

    client = InferenceClient(api_key=hf_token)

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct:together",
        messages=[{
            "role": "user",
            "content": prompt
            }
            ]
        )
    
    return response.choices[0].message.content

