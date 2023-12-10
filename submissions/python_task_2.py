import pandas as pd

def calculate_distance_matrix(dataset):
    df = pd.read_csv(dataset)
    distance_dict = {}
    # Iterate through rows and update the distance_dict
    for index, row in df.iterrows():
        source = int(row['id_start'])
        destination = int(row['id_end'])
        distance = float(row['distance'])
        # Update distance_dict with cumulative distances
        if source not in distance_dict:
            distance_dict[source] = {}
        if destination not in distance_dict:
            distance_dict[destination] = {}
        # Consider all possible routes and update the cumulative distances
        for intermediate in distance_dict[source].keys():
            if intermediate != destination:
                distance_dict[destination][intermediate] = distance_dict[source][intermediate] + distance
                distance_dict[intermediate][destination] = distance_dict[source][intermediate] + distance
        # Update the cumulative distance for the direct route
        distance_dict[source][destination] = distance
        distance_dict[destination][source] = distance
    # a DataFrame from the nested dictionary
    distance_matrix = pd.DataFrame.from_dict(distance_dict, orient='index')
    # fill NaN values with 0
    distance_matrix = distance_matrix.fillna(0)
    # Sorting index and columns
    distance_matrix.sort_index(axis=0, inplace=True)
    distance_matrix.sort_index(axis=1, inplace=True)
    # Ensuring the diagonal values are 0
    for col in distance_matrix.columns:
        distance_matrix.at[col, col] = 0.0
    return distance_matrix
dataset_path = '../datasets/dataset-3.csv'  # Replace with the actual path to your CSV file
result_distance_matrix = calculate_distance_matrix(dataset_path)

##Q2 task 2
def unroll_distance_matrix(distance_matrix):
    unrolled_data = []
    #traverse the rows of the distance_matrix
    for idx_start, row in distance_matrix.iterrows():
        for idx_end, distance in row.items():
            #exculde rows where id_start is equal to id_end
            if idx_start != idx_end:
                #append the data to the unrolled_data list
                unrolled_data.append({'id_start': idx_start, 'id_end': idx_end, 'distance': distance})
    #a DataFrame from the unrolled data
    unrolled_df = pd.DataFrame(unrolled_data)
    return unrolled_df
#result_distance_matrix is the DataFrame from the previous question
unrolled_distance_df = unroll_distance_matrix(result_distance_matrix)

#Q2---Task 3
def find_ids_within_ten_percentage_threshold(unrolled_distance_df, reference_value):
    # filtering rows based on the reference value in id_start column
    reference_rows = unrolled_distance_df[unrolled_distance_df['id_start'] == reference_value]
    # Calculate the average distance for the reference value
    reference_average_distance = reference_rows['distance'].mean()
    # Calculate the 10% threshold values
    lower_threshold = reference_average_distance * 0.9
    upper_threshold = reference_average_distance * 1.1
    # Filter rows within the 10% threshold
    within_threshold_rows = unrolled_distance_df[
        (unrolled_distance_df['distance'] >= lower_threshold) & (unrolled_distance_df['distance'] <= upper_threshold)
    ]
    # Get unique values from the id_start column and sort them
    result_ids = sorted(within_threshold_rows['id_start'].unique())
    return result_ids
#unrolled_distance_df is the DataFrame from the previous question
reference_value = 1001400
result_ids_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_distance_df, reference_value)

## Q2--Task 4
def calculate_toll_rate(distance_matrix):
    # Define rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Create new columns for each vehicle type and calculate toll rates
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        distance_matrix[vehicle_type] = distance_matrix['distance'] * rate_coefficient
    return distance_matrix
# Assuming distance_matrix is the DataFrame from the previous question
result_with_toll_rates = calculate_toll_rate(unrolled_distance_df)
result_with_toll_rates2=distance_matrix=result_with_toll_rates.drop('distance',axis=1)

##Q2 task 5
from datetime import time

def calculate_time_based_toll_rates(input_df):
    # Define rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Define time intervals within a day
    time_intervals = [
        {'start_time': time(0, 0), 'end_time': time(10, 0)},
        {'start_time': time(10, 0), 'end_time': time(18, 0)},
        {'start_time': time(18, 0), 'end_time': time(23, 59)},
    ]

    # Convert 'id_start' and 'id_end' columns to datetime
    input_df['id_start'] = pd.to_datetime(input_df['id_start'])
    input_df['id_end'] = pd.to_datetime(input_df['id_end'])

    # Create new columns for start_day, start_time, end_day, end_time
    input_df['start_day'] = input_df['id_start'].dt.day_name()
    input_df['start_time'] = input_df['id_start'].dt.time
    input_df['end_day'] = input_df['id_end'].dt.day_name()
    input_df['end_time'] = input_df['id_end'].dt.time

    # Apply the discount factors directly to the existing vehicle columns based on time intervals
    for interval in time_intervals:
        start_time = interval['start_time']
        end_time = interval['end_time']
        for vehicle_type, rate_coefficient in rate_coefficients.items():
            # Apply the discount factor to the existing vehicle column
            input_df[vehicle_type] = input_df.apply(lambda row: row[vehicle_type] * rate_coefficient if start_time <= row['start_time'] <= end_time and start_time <= row['end_time'] <= end_time else row[vehicle_type], axis=1)
        input_df=input_df[['id_start', 'id_end', 'distance', 'start_day', 'start_time', 'end_day', 'end_time',
                           'moto', 'car', 'rv', 'bus', 'truck']]
    return input_df

# Assuming unrolled_distance_df is the DataFrame from the previous question
result_with_time_based_toll_rates = calculate_time_based_toll_rates(unrolled_distance_df)


