import numpy as np
import math


M = 100
N = 1000


def load_trajectory_vector(file_path: str) -> np.ndarray:
    return np.load(file_path)


def get_njk_dict(trajectory_vector: np.ndarray, M) -> dict[int,dict[int,int]]:
    """Returns a dictionary of the amount of times (n) where we jump from j -> k
    Example: d = get_njk_dict([1,2,4,3,2,1,2]) = {
    1:{1:0, 2:2, 3:0, 4:0}, 
    2:{1:0, 2:0, 3:0, 4:1}, 
    3:{1:0, 2:1, 3:0, 4:0}, 
    4:{1:0, 2:0, 3:1, 4:0}
    }
    where d[j][k] = n
    """
    return_dict = {j:{
        k:0 for k in range(1, M+1)
    } for j in range(1, M+1)}

    for j_idx, k in enumerate(trajectory_vector[1:]):
        j = trajectory_vector[j_idx]
        return_dict[j][k] += 1
    
    return return_dict


def get_nj(n_jk:dict[int,dict[int,int]]):
    n_j = {key:0 for key in range(0,M+1)}
    for j, k_to_n_dict in n_jk.items():
        for k, n in k_to_n_dict.items():
            n_j[k] += n
    
    return n_j


def get_log_p_skatt_jk(n_jk, n_j, b, m) -> list[list[float]]:
    p = [[0 for _ in range(M)] for _ in range(M)]
    for j in range(1, m+1):
        for k in range(1, m+1):
            x1 = n_jk[j][k] + b
            x2 = n_j[j] + m*b
            p[j-1][k-1] += np.log(x1) - np.log(x2)

    return p


def get_logprod_p_skatt_jk(trajectory_vector, log_p_skatt_jk) -> float:
    s = 0
    for j_idx, k in enumerate(trajectory_vector[1:]):
        j = trajectory_vector[j_idx]
        s += log_p_skatt_jk[j-1][k-1] #zero indexed matrix
        
    return s


def compute_b_parameter_values(start, stop, step, n_jk, n_j, trajectory_data) -> list[(float, float)]:
    b_range = np.arange(start, stop, step)
    # Create candidate matrices
    p_jk_list = [get_log_p_skatt_jk(n_jk, n_j, b, M) for b in b_range]
    # Compute corresponding probability
    p_list = [get_logprod_p_skatt_jk(trajectory_data, p) for p in p_jk_list]
    b_to_p_list = [(b,p) for b, p in zip(b_range, p_list)]
    
    return b_to_p_list


def determine_optimal_b_parameter(start, stop, step, search_depth, depth_zoom, 
                                  n_jk, n_j, trajectory_data) -> float:
    current_start, current_stop, current_step = start, stop, step
    for depth in range(search_depth):
        b_to_p_list = compute_b_parameter_values(current_start, current_stop, current_step, n_jk, n_j, trajectory_data)
        b_index, (b_max, p_max) = max(enumerate(b_to_p_list), key=lambda x:x[1][1])
        current_start = b_max - current_step
        current_stop = b_max + current_step
        current_step /= depth_zoom
        print(b_max, p_max, depth)


def tests():
    assert get_njk_dict([1,2,4,3,2,1,2], 4)[1][2] == 2
    assert get_nj(get_njk_dict([1,2,4,3,2,1,2], 4))[2] == 3


def main():
    tests()

    trajectory_vector = load_trajectory_vector("trajectory.npy")
    assert N == len(trajectory_vector)

    trajectory_vec_1 = trajectory_vector[:N//2]
    trajectory_vec_2 = trajectory_vector[N//2:]
    assert len(trajectory_vec_1) == 500 and len(trajectory_vec_2) == 500

    n_jk = get_njk_dict(trajectory_vec_1, M)
    n_j = get_nj(n_jk)

    start = 0.000001
    stop = 2
    step = 0.01
    depth_zoom = 10
    search_depth = 6
    trajectory_data = trajectory_vec_2

    determine_optimal_b_parameter(start, stop, step, search_depth, depth_zoom, 
                                  n_jk, n_j, trajectory_data)

if __name__ == "__main__":
    main()