import numpy as np
from math import ceil
import time

def gen_fingerprints(fp_total, fp_size):
    """Generates random fingerprints.

    Arguments
    ---------
    fp_total : int
        Number of fingerprints.
    fp_size : int
        Size (or length) of the fingerprints.

    Returns
    -------
    total_fingerprints : np.array
        Numpy array containing the fingerprints.
    """
    return np.random.randint(2, size=(fp_total, fp_size), dtype='int8')

def calculate_counters(data_sets, c_threshold=None, w_factor="fraction"):
    """Calculate 1-similarity, 0-similarity, and dissimilarity counters

    Arguments
    ---------
    data_sets : np.ndarray
        Array of arrays. Each sub-array contains m + 1 elements,
        with m being the length of the fingerprints. The first
        m elements are the column sums of the matrix of fingerprints.
        The last element is the number of fingerprints.

    c_threshold : {None, 'dissimilar', int}
        Coincidence threshold.
        None : Default, c_threshold = n_fingerprints % 2
        'dissimilar' : c_threshold = ceil(n_fingerprints / 2)
        int : Integer number < n_fingerprints

    w_factor : {"fraction", "power_n"}
        Type of weight function that will be used.
        'fraction' : similarity = d[k]/n
                     dissimilarity = 1 - (d[k] - n_fingerprints % 2)/n_fingerprints
        'power_n' : similarity = n**-(n_fingerprints - d[k])
                    dissimilarity = n**-(d[k] - n_fingerprints % 2)
        other values : similarity = dissimilarity = 1

    Returns
    -------
    counters : dict
        Dictionary with the weighted and non-weighted counters.

    Notes
    -----
    Please, cite the original papers on the n-ary indices:
    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00505-3
    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00504-4
    """
    # Setting matches
    total_data = np.sum(data_sets, axis=0)
    n_fingerprints = int(total_data[-1])
    c_total = total_data[:-1]

    # Assign c_threshold
    if not c_threshold:
        c_threshold = n_fingerprints % 2
    if isinstance(c_threshold, str):
        if c_threshold != 'dissimilar':
            raise TypeError("c_threshold must be None, 'dissimilar', or an integer.")
        else:
            c_threshold = ceil(n_fingerprints / 2)
    if isinstance(c_threshold, int):
        if c_threshold >= n_fingerprints:
            raise ValueError("c_threshold cannot be equal or greater than n_fingerprints.")
        c_threshold = c_threshold

    # Set w_factor
    if w_factor:
        if "power" in w_factor:
            power = int(w_factor.split("_")[-1])

            def f_s(d):
                return power ** -float(n_fingerprints - d)

            def f_d(d):
                return power ** -float(d - n_fingerprints % 2)
        elif w_factor == "fraction":
            def f_s(d):
                return d / n_fingerprints

            def f_d(d):
                return 1 - (d - n_fingerprints % 2) / n_fingerprints
        else:
            def f_s(d):
                return 1

            def f_d(d):
                return 1
    else:
        def f_s(d):
            return 1

        def f_d(d):
            return 1

    # Calculate a, d, b + c
    a = 0
    w_a = 0
    d = 0
    w_d = 0
    total_dis = 0
    total_w_dis = 0
    for s in c_total:
        if 2 * s - n_fingerprints > c_threshold:
            a += 1
            w_a += f_s(2 * s - n_fingerprints)
        elif n_fingerprints - 2 * s > c_threshold:
            d += 1
            w_d += f_s(abs(2 * s - n_fingerprints))
        else:
            total_dis += 1
            total_w_dis += f_d(abs(2 * s - n_fingerprints))
    total_sim = a + d
    total_w_sim = w_a + w_d
    p = total_sim + total_dis
    w_p = total_w_sim + total_w_dis

    counters = {"a": a, "w_a": w_a, "d": d, "w_d": w_d,
                "total_sim": total_sim, "total_w_sim": total_w_sim,
                "total_dis": total_dis, "total_w_dis": total_w_dis,
                "p": p, "w_p": w_p}

    return counters
    
# Generating time benchmark results
# Repetitions
reps = 3

# Fingerprint size
fp_size = 167

# Fingerprint totals
fp_totals = range(1000000000, 10000000001, 1000000000)

# Generate random sets
# If True, all the fingerprints will be generated simultaneously.
# If False, we will generate the fingerprint matrix one column at a time.
simultaneously = True

time_list = []

if simultaneously:
    for fp_total in fp_totals:
        t = 0
        for rep in range(reps):
            total_fingerprints = gen_fingerprints(fp_total, fp_size)
            start = time.time()
            condensed_fingerprints = np.sum(total_fingerprints, axis=0)
            data_sets = np.array([np.append(condensed_fingerprints, fp_total)])
            counters = calculate_counters(data_sets)
            rr = counters["a"]/counters["p"]
            final_time = time.time() - start
            t += final_time
        time_list.append(t/reps)
else:
    for fp_total in fp_totals:
        t = 0
        for rep in range(reps):
            total_t = 0
            c_sums = []
            for i in range(fp_size):
                column = gen_fingerprints(fp_total, fp_size = 1)
                start = time.time()
                c_sums.append(column.sum(axis = 0)[0])
                total_t += time.time() - start
            start = time.time()
            c_sums.append(fp_total)
            data_sets = np.array([c_sums])
            counters = calculate_counters(data_sets)
            rr = counters["a"] / (counters["p"])
            total_t += time.time() - start
            t += total_t
        time_list.append(t/reps)

s = 'fp_total               time(s)\n'
for i in range(len(time_list)):
    s += '{:<10}  {:>18.6f}\n'.format(fp_totals[i], time_list[i])
with open('time_results.txt', 'w') as outfile:
    outfile.write(s)
