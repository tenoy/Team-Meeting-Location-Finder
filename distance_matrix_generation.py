from datetime import datetime
import googlemaps
import responses
from itertools import islice

# Script to query googles Distance-Matrix API
# Google distance matrix api allows only 25 origins OR destinations per request
# Also not more than 100 elements per request (number of elements = origins x destinations)
# Therefore the api is queried with 1 origin and a maximum of "chunksize" destinations
chunksize = 25 # chunksize of 25 results in queries with max. 25 elements

# Chunks a dict into dicts with a given chunksize "size"
def chunks(data, size):
    it = iter(data)
    for i in range(0, len(data), size):
        yield {k:data[k] for k in islice(it, size)}


# Setup api connection, query api and return dict of matrices
def query_google_api(destinations : dict[int, list], origins : dict[int, list], time_in_seconds : int, google_api_key : str) -> dict[tuple, list]:
    # Setup client
    key = google_api_key
    client = googlemaps.Client(key)
    # To speed up responses, dont retry to query if query limit is exceeded
    client.retry_over_query_limit = 0
    resp = responses.add(
                responses.GET, # Response method
                "https://maps.googleapis.com/maps/api/distancematrix/json", #url
                body='{"status":"OK","rows":[]}', #structure of body
                status=200, # http status code, 200 = OK
                content_type="application/json", # json format
            )

    # Store query results in matrices dict
    matrices = {}
    
    # For all starting locations
    for orig_key in origins:
        # Add station to origins list
        origs = []
        # Use coordinates for query (names might be ambiguous)
        origs.append({'lng': origins.get(orig_key)[0], 'lat': origins.get(orig_key)[1]})
        # The chunks function splits the data set into n data sets of size chunksize
        for chunk in chunks(destinations, chunksize):
            # Create a composite key that is composed of origin node number at index 0 and destination node numbers at indices 1 ... n
            composite_key = []
            composite_key.append(orig_key)
            # Add station to destinations list
            dests = []
            for dest_key in chunk:
                dests.append({'lng': destinations.get(dest_key)[0], 'lat': destinations.get(dest_key)[1]})
                composite_key.append(dest_key)
            
            # Query the api
            try:
                matrix = client.distance_matrix(origs, dests, mode="transit", transit_mode="train", arrival_time=time_in_seconds)
            except googlemaps.exceptions._OverQueryLimit:
                print("Query limit exceeded")
                matrices = {}
                return matrices
            except googlemaps.exceptions.Timeout:
                print("Timeout")
                matrices = {}
                return matrices
            else:
                # Gather result in matrices dict
                matrices.setdefault(tuple(composite_key), []).append(matrix)
    return matrices


# Converts google api response to data format that is used in Location Model class
def create_travel_time_matrix(destinations : dict[int, list], origins : dict[int, list], arrival_date : datetime, arrival_time : datetime, google_api_key : str) -> dict[int, list]:
    print("create_travel_time_matrix")
    print(arrival_date)
    print(arrival_time)
    arrival_date_time = datetime.combine(arrival_date, arrival_time)
    print(arrival_date_time)
    time_in_seconds = arrival_date_time.timestamp()
    print(time_in_seconds)

    matrices = query_google_api(destinations, origins, time_in_seconds, google_api_key)
    # Iterate of matrices and gather travel time durations in seconds
    travel_mat = {}
    if matrices == None:
        return travel_mat
    else:

        for key, value in matrices.items():
            origin = key[0]
            for item in value:
                for row in item.get('rows'):
                    # Index of key 0 = origin, 1..n = destinations
                    count = 1 
                    for element in row.get('elements'):
                        destination = key[count]
                        composite_key = []
                        # Destination must be first for optimization distance matrix
                        composite_key.append(destination) 
                        composite_key.append(origin)
                        # Get travel duration in seconds for responses with OK
                        if element.get('status') == 'OK':
                            travel_mat[tuple(composite_key)] = element.get('duration').get('value')
                        # For destination = origin, the api returns ZERO_RESULTS
                        if element.get('status') == 'ZERO_RESULTS':
                            if origin == destination:
                                travel_mat[tuple(composite_key)] = 0
                            else:
                                travel_mat[tuple(composite_key)] = 10**100
                        count += 1
        return travel_mat
