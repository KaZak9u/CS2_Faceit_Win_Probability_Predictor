from dotenv import load_dotenv
import os

load_dotenv()


API_KEY = os.getenv('API_KEY')
CS2_MM_ID = 'f4148ddd-bce8-41b8-9131-ee83afcdd6dd'
MY_ID = '7cd70802-9d25-4e17-8db5-248cc5e9ff84'
HEADERS = {
        'accept': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
ROPL_ID = '9bf535f6-dc33-498f-a5b2-66cb13c5476f'
