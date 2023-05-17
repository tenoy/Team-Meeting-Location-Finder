from itertools import product
from telnetlib import BINARY
from mip import Model, xsum, minimize
import geopy.distance
import distance_matrix_generation
import pandas as pd
import datetime

# Script that contains the optimization
class LocationModelOptimization:

    # Prepares data for optimization
    # start_trainstations is the set of all possible starting trainstations (about 6,5k) from which team members might start
    # preset_target_trainstations is the default set of possible trainstations, where the team might meet (about 20)
    # member_locations the actual selection of team member starting locations
    # excluded_locations target trainstations, that are excluded from optimization
    # arrival_date date with the date of the team meeting
    # arrival_time datetime with latest time of arrival to the team meeting
    def __init__(self, start_trainstations, preset_target_trainstations, member_locations, input_target_trainstations, excluded_locations, arrival_date, arrival_time, optimization_mode, google_api_key):
        print("Initialize optimization model")
        self.google_api_key = google_api_key
        self.start_trainstations = start_trainstations
        if (len(input_target_trainstations) > 1):
            self.target_trainstations = self.start_trainstations[self.start_trainstations['NAME'].isin(input_target_trainstations)]
        else:
            self.target_trainstations = preset_target_trainstations
        self.arrival_date = arrival_date
        self.arrival_time = arrival_time
        
        # Train stations
        self.names = {}
        self.facilities = []
        self.facility_names = []
        self.facility_positions = {}
        self.facility_capacities = {}
        # Member starting train stations
        self.customers = []
        self.customer_names = []
        self.customer_demands = {}
        self.customer_positions = {}
        test = self.target_trainstations["NAME"].tolist()
        # Index from iterrows starts at 0 and the first trainstation starts at row 2 in the data frame
        for index, row in self.start_trainstations.iterrows():
            self.names[index] = row["NAME"] #for overview
            if (row["NAME"] not in excluded_locations):
                if (row["NAME"] in self.target_trainstations["NAME"].tolist()):
                    self.facilities.append(index)
                    self.facility_positions[index] = (row["lng"], row["lat"])
                    self.facility_names.append(row["NAME"])
                    self.facility_capacities[index] = 10000
            if (row["NAME"] in member_locations):
                self.customers.append(index)
                self.customer_positions[index] = (row["lng"], row["lat"])
                self.customer_names.append(row["NAME"])
                self.customer_demands[index] = 1
        
        self.member_location_df = self.start_trainstations.loc[self.start_trainstations['NAME'].isin(member_locations)]
        self.dist = None
        self.best_location_name = None
        self.best_location_df = None
        self.travel_times_to_best_location_df = None
        if optimization_mode == 'Minimize total travel time':
            self.optimization_mode = 'minsum'
        else:
            self.optimization_mode = 'minmax'


    # Geopy uses (lat,lon) tuples to calc distance (method used for testing/development)
    def calcDist(self, lon1, lat1, lon2, lat2):
        coord_1 = (lat1, lon1) #switch the order from lon, lat in station data to lat, lon for calculation
        coord_2 = (lat2, lon2)
        distance = geopy.distance.distance(coord_1, coord_2).km
        return distance

     # 1-Median Problem Formulation (https://scipbook.readthedocs.io/en/latest/flp.html#the-k-median-problem)
    def medianModelSolution(self):
        print("Optimization using 1-Median Model started")
        m = Model()
        # Decision vars:
        # yi = 1 when facility i is opened, else 0
        y = {i: m.add_var(var_type=BINARY) for i in self.facilities}
        # xji = 1 when demand of customer j is met by facility i, else 0
        x = {(i, j): m.add_var(var_type=BINARY) for (i, j) in product(self.facilities, self.customers)}
        # Objective: Minimize total distance from facility to customers
        m.objective = minimize(
            xsum(self.dist[i, j] * x[i, j] for (i, j) in product(self.facilities, self.customers)))
        # Constraints:
        # Each customer j must be assigned to exactly 1 facility i
        for j in self.customers:
            m.add_constr(xsum(x[i, j] for i in self.facilities) == 1)
        # Exactly 1 facility must be opened
        m.add_constr(xsum(y[i] for i in self.facilities) == 1)
        # Add upper bound constraint
        for (i, j) in product(self.facilities, self.customers):
            m.add_constr(x[i, j] <= y[i])    
        # Customers can only be assigned to opened facilities 
        # -> this constraint makes all solutions infeasible & no comment on this restriction can be found in the source. 
        # Other sources dont have this constraint (e.g. Charikar & Guha, 2002)
        # for i in self.facilities:
        #     m0.add_constr(xsum(x0[i, j] for j in self.customers) <= y0[i])

        # Solve
        m.optimize(max_seconds=3)

        # That may be put into the solve_and_print method and this method just returns model m
        # However, obtaining vars from the model afterwards is only possible for integer models and is less comfortable
        index_best_location = -1
        if m.num_solutions:
            print("Solution with cost {} found.".format(m.objective_value))
            # print("Facilities capacities: {} ".format([z[f].x for f in self.facilities]))
            for (i, j) in [(i, j) for (i, j) in product(self.facilities, self.customers) if x[(i, j)].x >= 1e-6]:
                # print(str(i) + "," + str(j))
                # print(x[(i, j)].x)
                index_best_location = i
        print(f"Selected tainstation: {self.names[index_best_location ]} (Nr. {index_best_location})")
        print("Optimization using 1-Median Model done")
        return index_best_location

    # 1-Cover Problem Formulation with binary search https://scipbook.readthedocs.io/en/latest/flp.html#the-k-cover-problem
    # This is used instead of the 1-Center Problem Formulation, which has an computationally expensive min max objective function
    def coverModelSolution(self):
        print("Optimization using 1-Cover Model started")

        # Calculates the adjacency matrix for a given theta (duration/distance)
        def calcAdjacencyMatrix(self, theta):
            adjacencyMat = {}
            for (i, j) in product(self.facilities, self.customers):
                if self.dist[i, j] <= theta:
                    adjacencyMat[(i, j)] = 1
                else:
                    adjacencyMat[(i, j)] = 0
            return adjacencyMat
                
        # 1-Cover Problem Formulation and Solution
        def modelAndSolve(self, adjacencyMatrix):
            m = Model()
            # zj = 1 if customer j is NOT covered by any facility, else zj = 0
            z = {j: m.add_var(var_type=BINARY, name='z'+str(j)) for j in self.customers}
            # yi = 1 when facility i is opened, else 0
            y = {i: m.add_var(var_type=BINARY, name='y'+str(i)) for i in self.facilities}
            # Each customer j must be covered by Selected trainstation yi or must be indicated that customer j is not covered (zj = 1)
            for j in self.customers:
                m.add_constr(xsum(adjacencyMatrix[i, j] * y[i] for i in self.facilities) + z[j] >= 1)
            # The number of opened facilites must be 1    
            m.add_constr(xsum(y[i] for i in self.facilities) == 1)
            # Minimize the number of uncovered customers
            m.objective = minimize(
                xsum(z[j] for j in self.customers))
            m.optimize(max_seconds=3)
            if m.num_solutions:
                print("Solution for theta = {} with cost {} found.".format(theta, m.objective_value))
                for i in self.facilities:
                    if (y[i].x > 0):
                        print(f"Selected tainstation: {self.names[i]} (Nr. {i})")
                return m
            else:
                print("No Solution found for theta = {}".format(theta))
                print(m.objective_value)
                return m
        
        # Binary search procedure
        # Define lower bound and upper bound of travel time in (hours/mins/secs) for binary search (km for testing/development)
        # The idea of this method is to find a facility that covers all customers with min-max travel duration / distance
        lb = 0
        # Input roughly the double of longest duration (seconds) / distance (km) in germany: https://reisetopia.de/guides/laengste-zugstrecken-deutschlands/
        ub = 86400 #3000
        # Input the duration / distance at which we are indifferent regarding solution quality
        epsilon = 600 #10
        sol = None
        last_feasible_solution = None
        while ub - lb > epsilon:
            theta = (ub + lb) / 2
            adjacencyMatrix = calcAdjacencyMatrix(self, theta)
            sol = modelAndSolve(self, adjacencyMatrix)
            # Set ub to theta if all customers are covered by solution
            if sol.objective_value == 0:
                ub = theta
                # Only solutions that cover all customers are accepted
                last_feasible_solution = sol 
            else:
                lb = theta

        index_best_location = -1
        for i, v in enumerate(last_feasible_solution.vars):
            if v.x > 0:
                index_best_location = int(v.name[1:])

        print("Optimization using 1-Cover Model done")
        return index_best_location

    # Facility location problem formulation with SOS Type 1 vars (https://docs.python-mip.com/en/latest/examples.html)
    def facilityLocationModelSolution(self):
        print("Optimization using Facility Location Model with SOS Type1 started")        
        m = Model()
        # Plant capacity
        z = {i: m.add_var(ub=self.facility_capacities[i]) for i in self.facilities}  
        # Type 1 SOS (SOS = Special Ordered Sets, Type1 = Only 1 var may have val of 1 in this set) 
        # Only one plant per region -> only one train station in germany as target location
        # In this case, simply add all target trainstations since we dont have sub regions
        # Advantage vs. "normal" constraints is a faster search by the b&b alg
        for r in [0, 1]:
            # set of plants in region r
            Fr = [i for i in self.facilities if r * 50 <= self.facility_positions[i][0] <= 50 + r * 50]
            for i in self.facilities:
               if r * 50 <= self.facility_positions[i][0] <= 50 + r * 50:
                print(self.facility_positions[i][0])
                print(r)  
            m.add_sos([(z[i], i - 1) for i in Fr], 1)

        # Amount that plant i will supply to client j
        x = {(i, j): m.add_var() for (i, j) in product(self.facilities, self.customers)}

        # Satisfy demand
        for j in self.customers:
            m += xsum(x[(i, j)] for i in self.facilities) == self.customer_demands[j]

        # Plant capacity
        for i in self.facilities:
            m += z[i] >= xsum(x[(i, j)] for j in self.customers)

        # Objective function
        m.objective = minimize(
            xsum(self.dist[i, j] * x[i, j] for (i, j) in product(self.facilities, self.customers)))

        m.optimize(max_seconds=3)

        if m.num_solutions:
            print("Solution with cost {} found.".format(m.objective_value))
            # print("Facilities capacities: {} ".format([z[f].x for f in self.facilities]))

            index_best_location = -1
            # Print allocations
            for (i, j) in [(i, j) for (i, j) in product(self.facilities, self.customers) if x[(i, j)].x >= 1e-6]:
                print(str(i) + "," + str(j))
                print(x[(i, j)].x)
                index_best_location = i

        print("Optimization using Facility Location Model with SOS Type1 done")
        return index_best_location
        
    def solve_and_print(self):
        print("Travel Time Matrix generation and optimization procedure started")

        # Query Google Distance Matrix API to generate travel time matrix
        self.dist = distance_matrix_generation.create_travel_time_matrix(self.facility_positions, self.customer_positions, self.arrival_date, self.arrival_time, self.google_api_key)
        if len(self.dist) == 0:
            # For debugging / testing: generate (euclidean) distance matrix with geopy
            # for (i, j) in product(self.facilities, self.customers):
            #     self.dist[(i, j)] = self.calcDist(self.facility_positions[i][0], self.facility_positions[i][1], self.customer_positions[j][0], self.customer_positions[j][1])

            self.best_location_name = "Travel time matrix could not be created (Perhaps the Google Api Distance Matrix query limit is exceeded)"
            return
        index_best_location = -1
        if self.optimization_mode == 'minsum':
            index_best_location = self.medianModelSolution()
        else:
            index_best_location = self.coverModelSolution()
        
        self.best_location_name = self.names[index_best_location]
        self.best_location_df = self.target_trainstations.loc[self.target_trainstations['NAME'] == self.names[index_best_location]]

        traveltimes = {'Starting location' : [], 'Travel duration (hh:mm:ss)' : []}
        for i in self.customers:
            traveltimes['Starting location'].append(self.names[i])
            traveltimes['Travel duration (hh:mm:ss)'].append(str(datetime.timedelta(seconds = self.dist[index_best_location, i])))

        self.travel_times_to_best_location_df = pd.DataFrame(traveltimes)
        print('Travel Time Matrix generation and optimization procedure done')
