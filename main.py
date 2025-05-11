import numpy as np
import math


M = 100
N = 1000


def load_trajectory_vector(file_path: str) -> np.ndarray:
    return np.load(file_path)


def get_njk_dict(trajectory_vector: np.ndarray, M) -> dict[dict[int]]:
    """Returns a dictionary of the amount of times (n) where we jump from j -> k
    Example: d = get_njk_dict([1,2,4,3,2,1,2]) = {
    1:{1:0, 2:1, 3:0, 4:0}, 
    2:{1:2, 2:0, 3:1, 4:0}, 
    3:{1:0, 2:0, 3:0, 4:1}, 
    4:{1:0, 2:1, 3:0, 4:0}
    }
    where d[k][j] = n
    """
    return_dict = {k:{
        j:0 for j in range(1, M+1)
    } for k in range(1, M+1)}

    for j_idx, k in enumerate(trajectory_vector[1:]):
        j = trajectory_vector[j_idx]
        return_dict[k][j] += 1
    
    return return_dict


def get_nj(n_jk:dict[int:dict[int:int]]):
    n_j = dict()
    for key in n_jk.keys():
        n_j[key] = sum(n_jk[key].values())
    
    return n_j


def get_log_p_skatt_jk(n_jk, n_j, b, m) -> list[list[float]]:
    p = [[0 for _ in range(M)] for _ in range(M)]
    for j in range(1, m+1):
        for k in range(1, m+1):
            x1 = n_jk[k][j] + b
            x2 = n_j[j] + m*b
            p[j-1][k-1] += np.log(x1) - np.log(x2)

    return p


def get_logprod_p_skatt_jk(trajectory_vector, p_skatt_jk) -> float:
    s = 0
    for j_idx, k in enumerate(trajectory_vector[1:]):
        j = trajectory_vector[j_idx]
        s += p_skatt_jk[j-1][k-1]
        
    return s


def tests():
    assert get_njk_dict([1,2,4,3,2,1,2], 4)[2][1] == 2
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
    search_depth = 2

    p_jk_list = [get_log_p_skatt_jk(n_jk, n_j, b, M) for b in np.arange(start, stop, step)]
    p_list = [get_logprod_p_skatt_jk(trajectory_vec_2, p) for p in p_jk_list]

    x1, x2 = max(enumerate(p_list), key= lambda x:x[1])
    print(x2, start + x1 * step)

    for _ in range(search_depth):
        stop = x1*step + start + step
        start = x1*step + start - step
        step = step / 10

        p_jk_list = [get_log_p_skatt_jk(n_jk, n_j, b, M) for b in np.arange(start, stop, step)]
        p_list = [get_logprod_p_skatt_jk(trajectory_vec_2, p) for p in p_jk_list]
        x1, x2 = max(enumerate(p_list), key= lambda x:x[1])
        print(x2, start + x1 * step)


if __name__ == "__main__":
    main()