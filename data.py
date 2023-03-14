import requests
from dotenv import load_dotenv 

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

def geocode_address(address, GOOGLE_API_KEY):
    # API endpoint
    endpoint = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={GOOGLE_API_KEY}"

    # Make a request to the API
    response = requests.get(endpoint)

    # Check if the API response is successful
    if response.status_code == 200:
        # Parse the API response
        data = response.json()

        # Get the latitude and longitude from the API response
        latitude = data['results'][0]['geometry']['location']['lat']
        longitude = data['results'][0]['geometry']['location']['lng']

        return {'lat': latitude, 'lng': longitude}
    else:
        return None, None

# Example list of addresses
addresses = ['57 Sha Tsui Road, Tsuen Wan, Hong Kong',
                       '45 Kut Shing Street, Chai Wan, Hong Kong',
                       'Hong Kong Industrial Building, 444-452 Des Voeux Road West',
                       'Blue Box Factory Building, 25 Hing Wo Street, Tin Wan, Hong Kong',
                       'Yick Shiu Industrial Building, 1 San On Street, Tuen Mun, Hong Kong',
                       'Hung Wai Industrial Building, 3 Hi Yip Street, Yuen Long',
                       'Sun Tin Wai Estate, Sha Tin Tau Road, Sun Tin Wai, Hong Kong',
                       'Wing Cheung Industrial Building, Kwai Cheong Road, Kwai Chung, Hong Kong',
                       'Vigor Industrial Building, Cheung Tat Road, Tsing Yi, Hong Kong',
                       '13 Yip Cheong Street, Fanling, New Territories, Hong Kong',
                       'Jumbo Industrial Building, 189 Wai Yip Street, Kwun Tong, Kowloon, Hong Kong',
                       '20 Bute Street, Mong Kok, Hong Kong',
                       'Hang Fung Industrial Building, 2G Hok Yuen Street, Hung Hom, Hong Kong',
                       '106 King Fuk Street, San Po Kong, Kowloon, Hong Kong',
                       'Yee Kuk Industrial Centre, 555 Yee Kuk Street, Cheung Sha Wan, Kowloon, Hong Kong',
                       'Sunray Industrial Centre, 610 Cha Kwo Ling Road, Yau Tong, Hong Kong',
                       'On Ning Garden Block 2, 10 Sheung Ning Road, Hang Hau, Hong Kong'
                      ]

order_codes = ['','uGNhxg3peCUod72J','nfNuVc4JaGUTg57V','Joqb3n3CtmBrFLjb','tTU7RsyZir6sU5JS','Us4cyLoiPQSTCoKh',
              'iA2AZGB4CToJrAvN','e8V63e49aM8C53dQ','PEwHBxricjNrJfDA','xSAJpng7BqyGZeeW','c8yk5Sf7WfAtx5po',
              'om8oYUCK53xDnHmz','wMF9WpSTfzzuLuWU','Kfz2BVfZsNWycaca','EMswWvpnFXWeWpFQ','P5tnckgBeYmdFEWj',
              'np7BHyPwE88m46u4']

trackingIds = ['','2QXdqfx9w5','3jNJaXzNPr','7P6fa6P6yB','D9Edff4YWw','FMxtkEWabj','MtWNDFrf8R','bhwwR4sH2N',
              'em6CezZooK','fHUxm8Z63Y','fVcs7HLrPP','hXzQBBF6FW','jwkkjAFifU','qpZtesbvPz','t7bNoDCNr3',
              'toNjXJDqxW','xxEsy387NZ']

customer = [ \
  '',
  'Shanae Henderson', 
  'Macy Botterill', 
  'Stormi Matthews', 
  'Katelyn Marley', 
  'Mikey Pocock', 
  'Ryleigh Simmons', 
  'Lonny Huddleston', 
  'Jemima Black', 
  'Stacy Rowbottom', 
  'Jordin Rowe', 
  'Lolicia Baker', 
  'Romayne Whinery', 
  'Janice Hanson', 
  'Ryleigh Simmons', 
  'Philadelphia Becket', 
  'John Lacey'
]

routes = [[0, 4, 5, 9, 8, 0], [0, 2, 3, 15, 6, 0], [0, 13, 10, 16, 1, 0], [0, 7, 14, 11, 12, 0]]

# Geocode the addresses
coordinates = [geocode_address(address, GOOGLE_API_KEY) for address in addresses]

address_detail = {}
order_by_vehicle = {
    'vehicle 0': {}, 
    'vehicle 1': {}, 
    'vehicle 2': {}, 
    'vehicle 3': {}, 

}

data['demands'] = \
  [0, # depot
   1, 1, # 1, 2
   2, 4, # 3, 4
   2, 4, # 5, 6
   8, 8, # 7, 8
   1, 2, # 9,10
   1, 2, # 11,12
   4, 4, # 13, 14
   8, 8] # 15, 16

for order in range(len(customer)):
  address_detail[order] = {
      'id': order_codes[order],
      'courier': 0,
      'address': addresses[order],
      'mark': 0,
      'coordinates': coordinates[order],
      'trackingId': trackingIds[order]
  }

for vehicle in range(len(routes)):
  for destination in range(len(routes[vehicle])):
    if routes[vehicle][destination] != 0:
      address_detail[routes[vehicle][destination]]['courier'] = vehicle
      address_detail[routes[vehicle][destination]]['mark'] = routes[vehicle][destination]
    order_by_vehicle[f'vehicle {vehicle}'][f'destination {destination}'] = address_detail[routes[vehicle][destination]]
         


print(address_detail)
print(order_by_vehicle)