#Script to query googles Distance-Matrix API
from datetime import datetime
import googlemaps
from requests import Timeout
import responses
from itertools import islice


#Google distance matrix api allows only 25 origins OR destinations per request
#Therefore data sets must be partioned into chunks of n data sets with size of chunksize
chunksize = 25

#chunks a dict into dicts with a given chunksize "size"
def chunks(data, size):
    it = iter(data)
    for i in range(0, len(data), size):
        yield {k:data[k] for k in islice(it, size)}

#setup api connection, query api and return dict of matrices
def query_google_api(destinations : dict[int, list], origins : dict[int, list], time_in_seconds : int, google_api_key : str) -> dict[tuple, list]:
#Setup client
    key = google_api_key
    client = googlemaps.Client(key)
    #To speed up responses, dont retry to query if query limit is exceeded
    client.retry_over_query_limit = 0
    resp = responses.add(
                responses.GET, #response method
                "https://maps.googleapis.com/maps/api/distancematrix/json", #url
                body='{"status":"OK","rows":[]}', #structure of body
                status=200, #http status code, 200 = OK
                content_type="application/json", #json format
            )

    #store query results in matrices dict
    matrices = {}
    
    #For all starting locations
    for orig_key in origins:
        #add station to origins list
        origs = []
        #use coordinates for query (names might be ambiguous)
        origs.append({'lng': origins.get(orig_key)[0], 'lat': origins.get(orig_key)[1]})
        #the chunks function splits the data set into n data sets of size chunksize
        for chunk in chunks(destinations, chunksize):
            #create a composite key that is composed of origin node number at index 0 and destination node numbers at indices 1 ... n
            composite_key = []
            composite_key.append(orig_key)
            #add station to destinations list
            dests = []
            for dest_key in chunk:
                dests.append({'lng': destinations.get(dest_key)[0], 'lat': destinations.get(dest_key)[1]})
                composite_key.append(dest_key)
            
            #query the api
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
                #gather result in matrices dict
                matrices.setdefault(tuple(composite_key), []).append(matrix)
    #print(matrices)
    return matrices

#converts google api response to data format that is used in Location Model class
def create_travel_time_matrix(destinations : dict[int, list], origins : dict[int, list], arrival_date : datetime, arrival_time : datetime, google_api_key : str) -> dict[int, list]:
    print("create_travel_time_matrix")
    print(arrival_date)
    print(arrival_time)
    # print(time.mktime(time.strptime("13.07.2015 09:38:17", "%d.%m.%Y %H:%M:%S")))
    #print(time.timestamp())
    arrival_date_time = datetime.combine(arrival_date, arrival_time)
    #time_in_seconds = arrival_time.timestamp()
    print(arrival_date_time)
    time_in_seconds = arrival_date_time.timestamp()
    #time_in_seconds = datetime.timestamp(arrival_date)
    #time_in_seconds = arrival_time.second
    print(time_in_seconds)

    matrices = query_google_api(destinations, origins, time_in_seconds, google_api_key)
    #iterate of matrices and gather travel time durations in seconds
    travel_mat = {}
    if matrices == None:
        return travel_mat
    else:

        for key, value in matrices.items():
            #print(key)
            #print(value)
            origin = key[0]
            for item in value:
                #print(item)
                for row in item.get('rows'):
                    #print(row)
                    count = 1 #index of key 0 = origin, 1..n = destinations
                    for element in row.get('elements'):
                        #print(element)
                        destination = key[count]
                        composite_key = []
                        composite_key.append(destination) #destination must be first for optimization distance matrix
                        composite_key.append(origin)
                        #Get travel duration in seconds for responses with OK
                        if element.get('status') == 'OK':
                            #print(element.get('duration'))
                            travel_mat[tuple(composite_key)] = element.get('duration').get('value')
                        #For destination = origin, the api returns ZERO_RESULTS
                        if element.get('status') == 'ZERO_RESULTS':
                            if origin == destination:
                                travel_mat[tuple(composite_key)] = 0
                            else:
                                travel_mat[tuple(composite_key)] = 10**100
                        count += 1
        #print(travel_mat)
        return travel_mat
