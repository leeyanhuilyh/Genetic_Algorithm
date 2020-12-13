import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def generate_routes(cities_list,population_size):
    routes_list = []
    for i in range(population_size):
        routes_list.append(random.sample(cities_list,len(cities_list)))
    return routes_list

def calculate_distance(routes_list):
    distance_list = []
    fitness_list = []
    for i in range(len(routes_list)):
        distance = 0
        for j in range(num_cities):
            if j+1 > num_cities-1:
                distance += np.sqrt((routes_list[i][j][0]-routes_list[i][0][0])**2 + (routes_list[i][j][1]-routes_list[i][0][1])**2)
            else:
                distance += np.sqrt((routes_list[i][j][0]-routes_list[i][j+1][0])**2 + (routes_list[i][j][1]-routes_list[i][j+1][1])**2)
        distance_list.append(distance)
        fitness_list.append(1/distance)
    perc_list = pd.DataFrame({'Distance':distance_list,'Fitness':fitness_list}).sort_values(by=['Distance'])
    perc_list['cumsum'] = perc_list['Fitness'].cumsum()
    perc_list['cumperc'] = perc_list['cumsum']/max(perc_list['cumsum'])*100
    data = {'Routes': routes_list, 'Distance': distance_list, 'Perc': perc_list['cumperc']}
    routes_distance_df = pd.DataFrame(data).sort_values(by=['Distance'])
    return routes_distance_df

def selection(routes_distance_df, elite_size):
    selection_results = []
    routes_list = routes_distance_df['Routes'].tolist()
    elite_list = routes_list[:elite_size]
    for i in range(0, len(routes_list) - elite_size):
        pick = 100 * random.random()
        for i in range(0, len(routes_list)):
            if pick <= routes_distance_df.iat[i, 2]:
                selection_results.append(routes_list[i])
                break
    selection_results = elite_list + selection_results

    return selection_results

def breed_population(selection_results, elite_size):
    children = []
    routes_list = selection_results
    pool = random.sample(routes_list, len(routes_list))
    #elite_size = elite_size + 5

    for i in range(0, elite_size):
        children.append(routes_list[i])

    for i in range(0, len(routes_list) - elite_size):
        child = breed(pool[i], pool[len(routes_list) - i - 1])
        children.append(child)
    return children

def breed(parent1, parent2):
    child = []
    child_p1 = []
    child_p2 = []

    gene_a = int(random.random() * len(parent1))
    gene_b = int(random.random() * len(parent1))

    start_gene = min(gene_a, gene_b)
    end_gene = max(gene_a, gene_b)

    for i in range(start_gene, end_gene):
        child_p1.append(parent1[i])

    child_p2 = [item for item in parent2 if item not in child_p1]

    child = child_p1 + child_p2
    return child

def mutate_population(children, mutation_rate):
    mutated_pop = []
    for ind in range(len(children)):
        mutated_ind = mutate(children[ind], mutation_rate)  # len(population[ind]) is num_cities of x,y coords
        mutated_pop.append(mutated_ind)
    return mutated_pop   # get a list of individuals (mutated routes) in mutatedPop

def mutate(individual, mutation_rate):
    for swapped in range(len(individual)):  # individual is population[ind] and len = num_cities
        if (random.random() < mutation_rate): # mutationRate chance of mutation
            swap_with = int(random.random() * len(individual))
            city1 = individual[swapped]
            city2 = individual[swap_with]
            individual[swapped] = city2
            individual[swap_with] = city1
    return individual   # this individual would be a new route with swapped (mutated) cities

def random_local_search(routes_distance_df, elite_size):
    routes_list = routes_distance_df['Routes'].tolist()
    old_routes = routes_list[:-k]
    chosen_routes = random.sample(routes_list[:elite_size], k=k)
    reverse = []
    reverse_master = []
    randint1 = random.randint(1,num_cities-2)
    randint2 = random.randint(1,num_cities-2)
    while randint1 == randint2:
        randint2 = random.randint(1,num_cities-2)
    if randint1 < randint2:
        low = randint1
        high = randint2
    else:
        low = randint2
        high = randint1
    for route in chosen_routes:
        old_dist_low = np.sqrt((route[low-1][0]-route[low][0])**2 + (route[low-1][1]-route[low][1])**2)
        old_dist_high = np.sqrt((route[high][0]-route[high+1][0])**2 + (route[high][1]-route[high+1][1])**2)
        old_dist_total = old_dist_low + old_dist_high
        before = route[:low]
        after = route[high:]
        extracted = route[low:high]
        reverse = list(reversed(extracted))
        new_route = before + reverse + after
        new_dist_low = np.sqrt((new_route[low-1][0]-new_route[low][0])**2 + (new_route[low-1][1]-new_route[low][1])**2)
        new_dist_high = np.sqrt((new_route[high][0]-new_route[high+1][0])**2 + (new_route[high][1]-new_route[high+1][1])**2)
        new_dist_total = new_dist_low + new_dist_high
        if new_dist_total < old_dist_total:
            reverse_master.append(new_route)
        else:
            reverse_master.append(route)
    routes_list = reverse_master + old_routes
    return routes_list

def new_generation(routes_list):
    routes_distance_df = calculate_distance(routes_list) # routes_distance_df
    routes_list = random_local_search(routes_distance_df, elite_size)
    routes_distance_df = calculate_distance(routes_list)
    selection_results = selection(routes_distance_df, elite_size) # type(selection) = list of routes randomly picked by fitness
    children = breed_population(selection_results, elite_size) # type(children) = list
    new_generation_list = mutate_population(children, mutation_rate) # type(children) = list
    new_generation_df = calculate_distance(new_generation_list)
    return new_generation_df

def plot_graphs(best_distances,saved_route,distance_history):
    plt.figure(1)
    plt.suptitle("Best Route")
    best_route_x = []
    best_route_y = []
    for point in saved_route:
        best_route_x.append(point[0])
        best_route_y.append(point[1])
    best_route_x.append(saved_route[0][0])
    best_route_y.append(saved_route[0][1])
    distance = 0
    for i in range(len(saved_route)):
        if i + 1 > len(saved_route) - 1:
            distance += np.sqrt((saved_route[i][0] - saved_route[0][0]) ** 2 + (saved_route[i][1] - saved_route[0][1]) ** 2)
        else:
            distance += np.sqrt((saved_route[i][0] - saved_route[i+1][0]) ** 2 + (saved_route[i][1] - saved_route[i+1][1]) ** 2)
    plt.title(distance)
    plt.scatter(best_route_x,best_route_y)
    plt.plot(best_route_x,best_route_y)

    plt.figure(2)
    plt.plot(best_distances)
    plt.ylabel('Distance')
    plt.xlabel('Generation')

    plt.figure(3)
    plt.plot(distance_history)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()

def select_points(num_cities):
    def notify(s):
        plt.title(s, fontsize=12)
        plt.draw()

    num_pts = num_cities
    plt.clf()
    notify("Please plot your points")
    plt.waitforbuttonpress()
    notify("")
    pts_x = []
    pts_y = []
    cities_list = []
    ptsPlotted = False
    while ptsPlotted == False:
        pts = []
        while len(pts) < num_pts:
            pts = np.asarray(plt.ginput(num_pts, timeout=-1))
            if len(pts) < num_pts:
                notify('Too few points, starting over')
                time.sleep(1)  # Wait a second
        ptsPlotted = True

    for pt in pts:
        pts_x.append(pt[0]*100)
        pts_y.append(pt[1]*100)

    plt.figure(0)
    plt.suptitle("Initial points selected")
    plt.scatter(pts_x, pts_y)
    plt.show(block=False)

    for pt in pts:
        cities_list.append((pt[0]*100,pt[1]*100))

    return cities_list

def benchmark_points():
    benchmark = 'data/eil76.txt'  # att48, berlin52, chn31, chn144, eil76, gr96
    cities = np.loadtxt(benchmark)
    cities_list = []
    for line in cities:
        cities_list.append((line[1], line[2]))
    return cities_list

### initialise global variables ###
population_size = 20
elite_size = 10
mutation_rate = 0.0001
generations = 500
k = 8 # for local_search

cities_list = select_points(20) # either use select_points(number of pts) or benchmark_points()
start = time.time()
num_cities = len(cities_list)
routes_list = generate_routes(cities_list, population_size)
routes_list_df = new_generation(routes_list)
best_distances = []
distance_history = []
for i in range(generations):
    routes_list_df = new_generation(routes_list_df['Routes'].tolist())
    distance_history.append(routes_list_df['Distance'].tolist()[0])
    if i == 0 or routes_list_df['Distance'][0] < best_distances[-1]:
        best_distances.append(routes_list_df['Distance'][0])
        saved_route = []
        for point in (routes_list_df['Routes'][0]):
            saved_route.append(point)
    else:
        best_distances.append(best_distances[-1])
end = time.time()
print(f'Runtime: {end-start}')
plot_graphs(best_distances,saved_route,distance_history)
