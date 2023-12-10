import pandas as pd

## Q1---task1
def generate_car_matrix(dataset):
    df = pd.read_csv(dataset)
    # Pivot the DataFrame using id_1, id_2, and car columns
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car')
    # Fill NaN values with 0 (since diagonal values should be 0)
    car_matrix = car_matrix.fillna(0)
    #returning the new-dataframe
    return car_matrix
dataset_path = '../datasets/dataset-1.csv'
result_matrix = generate_car_matrix(dataset_path)


## Q1---- Task 2
def get_type_count(dataset):
    data=pd.read_csv(dataset)
    data['car_type']=data.car.apply(lambda x:'low' if x<=15.0 else('medium' if x>15.0 and x<=25.0 else 'high'))
    dictionary=data['car_type'].value_counts().to_dict()
    sorted_type_count = dict(sorted(dictionary.items()))
    return sorted_type_count
new_dataset=get_type_count(dataset_path)

##Q3---Task 3
import pandas as pd

def get_bus_indexes(dataset):
    df = pd.read_csv(dataset)
    mean_bus_value = df['bus'].mean() # Calculating the mean value of the 'bus' column
    # Identifying indices where the 'bus' values are greater than twice the mean of bus
    bus_indexes = df[df['bus'] > 2 * mean_bus_value].index.tolist()
    bus_indexes.sort() # Sorting the indices in ascending order
    return bus_indexes
dataset_path = '../datasets/dataset-1.csv'
dataset_path = dataset_path
sorted_bus_indexes = get_bus_indexes(dataset_path)

##Q1---Task4
def filter_routes(dataset):
    df = pd.read_csv(dataset)
    # Calculate the average value of the 'truck' column for each unique 'route'
    route_avg_truck = df.groupby('route')['truck'].mean()
    # Filter routes where the average of 'truck' values is greater than 7
    selected_routes = route_avg_truck[route_avg_truck > 7].index.tolist()
    selected_routes.sort() # Sort the list of selected routes
    return selected_routes
dataset_path = '../datasets/dataset-1.csv'  # Replace with the actual path to your CSV file
result_routes = filter_routes(dataset_path)

##Q1--Task5
def multiply_matrix(car_matrix):
    # Apply the modification logic to each element in the DataFrame
    modified_matrix = car_matrix.map(lambda x: x * 0.75 if x > 20 else x * 1.25)
    modified_matrix = modified_matrix.round(1)# Round the values to 1 decimal place
    return modified_matrix
#DataFrame from Question 1
result_modified = multiply_matrix(result_matrix)



