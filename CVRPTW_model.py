# [START import]
from functools import partial
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np

# [END import]

import requests
import json
import urllib
from dotenv import load_dotenv 

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

def create_data():
  """Creates the data."""
  data = {}
  data['API_key'] = GOOGLE_API_KEY
  data['addresses'] = ['57 Sha Tsui Road, Tsuen Wan, Hong Kong',
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
  for i in range(len(data['addresses'])):
    data['addresses'][i] = data['addresses'][i].replace(" ", "+" )

  return data

def create_distance_matrix(data):
  addresses = data["addresses"]
  API_key = data["API_key"]
  # Distance Matrix API only accepts 100 elements per request, so get rows in multiple requests.
  max_elements = 100
  num_addresses = len(addresses) # 16 in this example.
  # Maximum number of rows that can be computed per request (6 in this example).
  max_rows = max_elements // num_addresses
  # num_addresses = q * max_rows + r (q = 2 and r = 4 in this example).
  q, r = divmod(num_addresses, max_rows)
  dest_addresses = addresses
  distance_matrix = []
  time_matrix = []
  # Send q requests, returning max_rows rows per request.
  for i in range(q):
    origin_addresses = addresses[i * max_rows: (i + 1) * max_rows]
    response = send_request(origin_addresses, dest_addresses, API_key)
    distance_matrix += build_distance_matrix(response)
    time_matrix += build_time_matrix(response)


  # Get the remaining remaining r rows, if necessary.
  if r > 0:
    origin_addresses = addresses[q * max_rows: q * max_rows + r]
    response = send_request(origin_addresses, dest_addresses, API_key)
    distance_matrix += build_distance_matrix(response)
    time_matrix += build_time_matrix(response)

  big_matrix = {"distance_matrix": distance_matrix, "time_matrix": time_matrix}

  return big_matrix

def send_request(origin_addresses, dest_addresses, API_key):
  """ Build and send request for the given origin and destination addresses."""
  def build_address_str(addresses):
    # Build a pipe-separated string of addresses
    address_str = ''
    for i in range(len(addresses) - 1):
      address_str += addresses[i] + '|'
    address_str += addresses[-1]
    return address_str

  request = 'https://maps.googleapis.com/maps/api/distancematrix/json?units=imperial'
  origin_address_str = build_address_str(origin_addresses)
  dest_address_str = build_address_str(dest_addresses)
  request = request + '&origins=' + origin_address_str + '&destinations=' + \
                       dest_address_str + '&key=' + API_key
  jsonResult = urllib.request.urlopen(request).read()
  response = json.loads(jsonResult)
  return response

def build_distance_matrix(response):
  distance_matrix = []
  for row in response['rows']:
    row_list = [round(row['elements'][j]['distance']['value']/1000) for j in range(len(row['elements']))]
    distance_matrix.append(row_list)
  return distance_matrix

def build_time_matrix(response):
  time_matrix = []
  for row in response['rows']:
    row_list = [round(row['elements'][j]['duration']['value']/60) for j in range(len(row['elements']))]
    time_matrix.append(row_list)
  return time_matrix

########
# Main #
########
def data_model():
  """Entry point of the program"""
  # Create the data.
  data = create_data()
  addresses = data['addresses']
  API_key = data['API_key']
  distance_matrix = create_distance_matrix(data)
  return distance_matrix

# [START data_model]
def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['time_windows'] = \
        [
          (0, 100),  # depot
          (7, 100),  # 1
          (10, 100),  # 2
          (16, 100),  # 3
          (10, 100),  # 4
          (0,100),  # 5
          (5, 100),  # 6
          (0,100),  # 7
          (5, 100),  # 8
          (0,100),  # 9
          (10, 100),  # 10
          (10, 100),  # 11
          (0,100),  # 12
          (5, 100),  # 13
          (7,100),  # 14
          (10, 100),  # 15
          (11, 100),  # 16
        ]

    data['distance_matrix'] = data_model()['distance_matrix']

    data['time_matrix'] = data_model()['time_matrix']
    
    data['num_locations'] = len(data['distance_matrix'])

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
    # data['time_per_demand_unit'] = 5  # 5 minutes/unit
    data['num_vehicles'] = 4
    data['vehicle_capacity'] = 15
    # data['vehicle_speed'] = 83  # Travel speed: 5km/h converted in m/min
    data['depot'] = 0
    return data
    # [END data_model]


#######################
# Problem Constraints #
#######################


def create_time_evaluator(data):
    """Creates callback to get total times between locations."""

    def service_time(data, node):
        """Gets the service time for the specified location."""
        return data['demands'][node] * data['time_per_demand_unit']

    def travel_time(data, from_node, to_node):
        """Gets the travel times between two locations."""
        if from_node == to_node:
            travel_time = 0
        else:
            travel_time = data['distance_matrix'][from_node][to_node] / data['vehicle_speed']
        return travel_time

    _total_time = {}
    # precompute total time to have time callback in O(1)
    for from_node in range(data['num_locations']):
        _total_time[from_node] = {}
        for to_node in range(data['num_locations']):
            if from_node == to_node:
                _total_time[from_node][to_node] = 0
            else:
                _total_time[from_node][to_node] = int(
                    service_time(data, from_node) + travel_time(
                        data, from_node, to_node))
                
    print(_total_time)
    def time_evaluator(manager, from_node, to_node):
        """Returns the total time between the two nodes"""
        return _total_time[manager.IndexToNode(from_node)][manager.IndexToNode(
            to_node)]

    return time_evaluator    

def get_routes(data, manager, routing, solution):
  """Get vehicle routes from a solution and store them in an array."""
  # Get vehicle routes and store them in a two dimensional array whose
  # i,j entry is the jth location visited by vehicle i along its route.
  routes = []
  for route_nbr in range(routing.vehicles()):
    index = routing.Start(route_nbr)
    route = [manager.IndexToNode(index)]
    while not routing.IsEnd(index):
      index = solution.Value(routing.NextVar(index))
      route.append(manager.IndexToNode(index))
    routes.append(route)
  data['route'] = routes
  print(len(data['route']))
  return routes

# [START solution_printer]
def print_solution(manager, routing, assignment):  # pylint:disable=too-many-locals
    # route = np.zero(data)
    """Prints assignment on console"""
    print(f'Objective: {assignment.ObjectiveValue()}')
    time_dimension = routing.GetDimensionOrDie('Time')
    capacity_dimension = routing.GetDimensionOrDie('Capacity')
    total_distance = 0
    total_load = 0
    total_time = 0
    print(manager.GetNumberOfVehicles(), assignment)

    for vehicle_id in range(manager.GetNumberOfVehicles()):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        distance = 0
        while not routing.IsEnd(index):
            load_var = capacity_dimension.CumulVar(index)
            time_var = time_dimension.CumulVar(index)
            slack_var = time_dimension.SlackVar(index)
            plan_output += ' {0} Load({1}) Time({2},{3}) Slack({4},{5}) ->'.format(
                manager.IndexToNode(index),
                assignment.Value(load_var),
                assignment.Min(time_var),
                assignment.Max(time_var),
                assignment.Min(slack_var), assignment.Max(slack_var))
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            distance += routing.GetArcCostForVehicle(previous_index, index,
                                                     vehicle_id)
        load_var = capacity_dimension.CumulVar(index)
        time_var = time_dimension.CumulVar(index)
        slack_var = time_dimension.SlackVar(index)
        plan_output += ' {0} Load({1}) Time({2},{3})\n'.format(
            manager.IndexToNode(index),
            assignment.Value(load_var),
            assignment.Min(time_var), assignment.Max(time_var))
        plan_output += 'Distance of the route: {0}km\n'.format(distance)
        plan_output += 'Load of the route: {}\n'.format(
            assignment.Value(load_var))
        plan_output += 'Time of the route: {0}hour {1}mins\n'.format(
            assignment.Value(time_var)//60, 
            assignment.Value(time_var)%60) if assignment.Value(time_var) >= 60 else 'Time of the route: {}mins\n'.format(
            assignment.Value(time_var))
        print(plan_output)
        total_distance += distance
        total_load += assignment.Value(load_var)
        total_time += assignment.Value(time_var)
    print('Total Distance of all routes: {0}km'.format(total_distance))
    print('Total Load of all routes: {}'.format(total_load))
    print('Total Time of all routes: {0}hours {1}mins\n'.format(
            total_time//60, 
            total_time%60) if total_time >= 60 else 'Total Time of all routes: {}mins\n'.format(
            total_time))
    # [END solution_printer]


def main():
    """Solve the Capacitated VRP with time windows."""
    # Instantiate the data problem.
    # [START data]
    data = create_data_model()
    # [END data]

    # Create the routing index manager.
    # [START index_manager]
    manager = pywrapcp.RoutingIndexManager(data['num_locations'],
                                           data['num_vehicles'], data['depot'])
    # [END index_manager]

    # Create Routing Model.
    # [START routing_model]
    routing = pywrapcp.RoutingModel(manager)
    # [END routing_model]

    # Define weight of each edge.
    # [START transit_callback]
    def distance_evaluator(from_node, to_node):
      """Creates callback to return distance between points."""
      """Returns the manhattan distance between the two nodes"""
      from_node = manager.IndexToNode(from_node)
      to_node = manager.IndexToNode(to_node)
      return data['distance_matrix'][from_node][to_node]

    distance_evaluator_index = routing.RegisterTransitCallback(distance_evaluator)
    # [END transit_callback]

    # Define cost of each arc.
    # [START arc_cost]
    routing.SetArcCostEvaluatorOfAllVehicles(distance_evaluator_index)
    # [END arc_cost]

    # Add Capacity constraint.

    def demand_evaluator(node):
        """Creates callback to get demands at each location."""

        from_node = manager.IndexToNode(node)

        return data['demands'][from_node]

        
    # [START capacity_constraint]
    demand_evaluator_index = routing.RegisterUnaryTransitCallback(demand_evaluator)
    routing.AddDimension(
        demand_evaluator_index,
        0,  # null capacity slack
        data['vehicle_capacity'],
        True,  # start cumul to zero
        'Capacity'
    )
    # [END capacity_constraint]

    # Add Time Window constraint.

    def time_evaluator(from_node, to_node):
        """Returns the total time between the two nodes"""
        from_node = manager.IndexToNode(from_node)
        to_node = manager.IndexToNode(to_node)
        return data['time_matrix'][from_node][to_node]

    # [START time_constraint]
    time_evaluator_index = routing.RegisterTransitCallback(time_evaluator)

    """Add Global Span constraint"""
    routing.AddDimension(
        time_evaluator_index,
        120,  # allow waiting time
        120,  # maximum time per vehicle
        False,  # don't force start cumul to zero since we are giving TW to start nodes
        'Time'
    )
    time_dimension = routing.GetDimensionOrDie('Time')
    # Add time window constraints for each location except depot
    # and 'copy' the slack var in the solution object (aka Assignment) to print it
    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx == data['depot']:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
        routing.AddToAssignment(time_dimension.SlackVar(index))
    # Add time window constraints for each vehicle start node
    # and 'copy' the slack var in the solution object (aka Assignment) to print it
    depot_idx = data['depot']
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(
            data['time_windows'][0][0],
            data['time_windows'][0][1])
        routing.AddToAssignment(time_dimension.SlackVar(index))
        # Warning: Slack var is not defined for vehicle's end node
        #routing.AddToAssignment(time_dimension.SlackVar(self.routing.End(vehicle_id)))

    # [END time_constraint]

    # Setting first solution heuristic (cheapest addition).
    # [START parameters]
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.FromSeconds(2)
    search_parameters.log_search = True
    # [END parameters]

    # Solve the problem.
    # [START solve]
    solution = routing.SolveWithParameters(search_parameters)
    # [END solve]

    # Print solution on console.
    # [START print_solution]
    if solution:
        print_solution(manager, routing, solution)
        print(get_routes(data, manager, routing, solution))
    else:
        print('No solution found!')
    # [END print_solution]

    print(data)

if __name__ == '__main__':
    main()