import requests 
from dotenv import load_dotenv 

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')


def extract_lat_long_via_address(address_or_zipcode):
    lat, lng = None, None
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    endpoint = f"{base_url}?address={address_or_zipcode}&key={GOOGLE_API_KEY}"
    # see how our endpoint includes our API key? Yes this is yet another reason to restrict the key
    r = requests.get(endpoint)
    if r.status_code not in range(200, 299):
        return None, None
    try:
        '''
        This try block incase any of our inputs are invalid. This is done instead
        of actually writing out handlers for all kinds of responses.
        '''
        results = r.json()['results'][0]
        lat = results['geometry']['location']['lat']
        lng = results['geometry']['location']['lng']
    except:
        pass
    print(lat, lng)
    return lat, lng

if __name__ == "__main__":
  extract_lat_long_via_address('57 Sha Tsui Road, Tsuen Wan, Hong Kong')