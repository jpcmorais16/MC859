import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize


def read_data(filename, variables=[], results=[]):
    with open(filename, 'r') as file:
        
        i = 1
        probability_step_length_one = .0
        n_nodes = 0
        n_marked = 0
        n_distant_edges = 0
        gama = 0

        average_clustering = 0
        transitivity = 0
        average_shortest_path_length = 0
        number_of_edges = 0
        exponent = 0
        
        for line in file:
            if i % 14 == 2:
                values = line.split(" ")
                probability_step_length_one = float(values[0])
                n_nodes = float(values[1])
                n_marked = float(values[2])
                n_distant_edges = float(values[3])
                gama = float(values[4])

                variables.append((probability_step_length_one, n_nodes, n_marked, n_distant_edges, gama))
    
            if i % 14 == 8: #average clustering
                values = line.split(": ")
                average_clustering = float(values[1][:-1])
            
            if i % 14 == 9: #transitivity
                values = line.split(": ")
                transitivity = float(values[1][:-1])

            if i % 14 == 10: #average shortest path length
                values = line.split(": ")
                average_shortest_path_length = float(values[1][:-1])

            if i % 14 == 11:
                values = line.split(": ")
                number_of_edges = float(values[1][:-1])

            if i % 14 == 12:
                values = line.split(": ")
                exponent = float(values[1][:-1])
            
            if i % 14 == 0:
                results.append((average_clustering, transitivity, average_shortest_path_length, number_of_edges))

            i += 1

    return variables, results

variables = []
results = []

for filename in ["file1.txt", "file2.txt", "file3.txt", "file4.txt", "file5.txt", "file6.txt", "file7.txt", "file8.txt", "file9.txt", "file10.txt"]:

    variables, results = read_data(filename, variables, results)


list = [(variables[i], results[i]) for i in range(len(variables)) if variables[i][3] >= 1]
# list_11k = [(variables[i], results[i]) for i in range(len(variables)) if variables[i][1] == 11000 and 3.8 <= results[i][2] <= 4.1]
# list_21k = [(variables[i], results[i]) for i in range(len(variables)) if variables[i][1] == 21000 and 4.2 <= results[i][2] <= 4.5]
list_11k = [(variables[i], results[i]) for i in range(len(variables)) if variables[i][1] == 11000 and variables[i][2] == 7 and variables[i][3] == 1]
list_21k = [(variables[i], results[i]) for i in range(len(variables)) if variables[i][1] == 21000 and variables[i][2] == 7 and variables[i][3] == 1]

m = sorted(list, key=lambda x: (x[1][0])*x[1][1]/(x[1][2]**1.5), reverse=True)
m11k = sorted(list_11k, key=lambda x: (x[1][0])*x[1][1], reverse=True)
m21k = sorted(list_21k, key=lambda x: (x[1][0])*x[1][1], reverse=True)

for j in m[:5]:

    print(j)

for j in m11k[:5]:

    print(j)

for j in m21k[:5]:

    print(j)


num_samples = 1000


# Dividir em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(results, variables, test_size=0.01)

# Treinar um modelo de regressão para aproximar a função inversa
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Testar o modelo
example_output = np.array([0.67, 0.5, 4, 50000])  # Um exemplo de saída da função
predicted_input = model.predict([example_output])

print("Resultado esperado:", example_output)
print("Inputs preditos:", predicted_input)