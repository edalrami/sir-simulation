import numpy as np
import pandas as pd
from tqdm import tqdm

def generate_initial_point(low, high, n, infected_idx, default_val=1):
    """ generaete initial point

    Parameters: low (int/float): lower boundary
                high (int/float): upper boundary
                n (int): total points
                infected_idx (dict) - index of those asymptomatic or infected or recovered
                                      format -> {id/idx:status}
                default_val (int) - default value if not specified in infected_idx, optional, default is 1

    Returns: init_point (DataFrame) - a dataframe with 3 columns: x, y, status
    """
    # generate walk
    init_point = pd.DataFrame(np.random.uniform(low, high, n*2).reshape((n,2)), columns=['x','y'])
    init_point['status'] = [infected_idx.get(i, default_val) for i in range(n)]
    return init_point

def single_pt_infection(in_array, in_pt, threshold):
    """ calculate the distance between a nd array with a single point
        and determine if it's True or False by the threshold

    Parameters: in_array (DataFrame) - the points, has to be a dataframe for preserving the index
                in_pt (Series) - the single point
                threshold (int/float) - the threshold to determine if it's True or not

    Returns: result (Series) - a series of boolean values with the original index preserved
    """
    dist = np.sum((in_array - in_pt)**2, 1)**0.5
    result = dist <= threshold
    return result

def update_status_by_location(in_df, infection, non_im, threshold):
    """ update the status if the distance between a point and an infected point is less than a threshold

    Parameters: in_df (DataFrame) - the dataframe contains the x,y and status data
                infection (array-like) - code denotes that is affected, eg [2,3]
                non_im_df (array-like) - code denotes people with no immunity, eg [1]
                threshold (int/float) - threshold distance

    Returns: return_df (DataFrame) - the updated status data
    """
    return_df = in_df.copy()
    infection_df = return_df[return_df.status.isin(infection)]
    non_im_df = return_df[return_df.status.isin(non_im)]

    # healthy people being affected or not
    affect_df_result = pd.DataFrame([single_pt_infection(non_im_df[['x','y']], j, threshold) for i,j in infection_df.iterrows()]).T
    affect_idx = affect_df_result[affect_df_result.sum(1) > 0].index

    # change return_df status column
    for i in affect_idx:
        return_df.at[i,'status'] = 2

    return return_df

def random_walk(in_df, low, high):
    """ input a data and generate random walk base on a low and high value

    Parameters: in_df (DataFrame) - input dataframe with 'x', 'y' column
                low (int/float) - the lower bound
                high (int/float) - the upper bound

    Returns: return_df (DataFrame) - the updated x,y data
    """
    dim = in_df.shape[0]
    new_pts = np.array([np.random.uniform(i-low, i+high) for i in in_df[['x', 'y']].values.flatten()])
    new_pts = new_pts.reshape((dim,2))
    new_pts_df = pd.DataFrame(new_pts, columns=['x','y'])
    return_df = pd.concat([new_pts_df, in_df[[i for i in list(in_df) if i not in ['x','y']]]], 1)

    return return_df

def distance_check_single(in_df, in_point, exclude, threshold):
    """ check if a single point has a larger distance than all other points
    """
    in_array = in_df[['x','y']].values
    all_dist = np.sum((in_df - in_point)**2, 1)**0.5
    all_dist = np.array([j for i,j in enumerate(all_dist) if i != exclude])
    return all(all_dist >= threshold)

def distance_check(in_df, threshold):
    return all([distance_check_single(in_df, j, i, threshold) for i,j in in_df.iterrows()])

def social_distancing(in_df, low, high, threshold):
    """ generate data with random walk function which all data point
        must maintain a distance with other larger than a threshold
    """
    while True:
        new_data = random_walk(in_df, low, high)
        if distance_check(new_data, threshold):
            break

    return new_data

## main function ##

def simulation(low, high, n, infected_idx, total_step, infection, non_im, infection_threshold, rw_low, rw_high, sd_threshold, default_val=1):
    """ simulation

    Parameters: low (int/float) - lower bound for initial random generation
                high (int/float) - upper bound for initial random generation
                n (int) - how many people in this simulation
                infected_idx (dict) - infected id and type in a dictionary, {id/index:type (1-4)}
                total_step (int) - total simulation steps
                infection (list) - infected, i.e. [2,3] in our case
                non_im (list) - non-immunity, i.e. [4] in our case
                infection_threshold (int/float) - will be infected within this distance values from anyone
                                                  has status code specified in "infection"
                rw_low (int/float) - random walk lower bound from the previous location
                rw_high (int/float) - random walk upper bound from the previous location
                sd_threshold (int/float) - minimum social distancing value
                default_val (int) - values for whatever code is not specified in infected_idx

    Returns: step_record (dict) - a dictionary contains record of each step, {step_no:dataframe}
    """
    step_record = dict()
    # start with a random data
    initial_data = generate_initial_point(low, high, n, infected_idx, default_val)
    initial_data['step'] = [0]*n
    step_record[0] = initial_data

    for i in tqdm(range(1, total_step+1)):
        # update infection status
        new_data = update_status_by_location(initial_data, infection, non_im, infection_threshold)
        new_data['step'] = [i]*n
        step_record[i] = new_data
        # random walk with social distancing
        new_data = social_distancing(new_data, rw_low, rw_high, sd_threshold)
        initial_data = new_data

    return step_record