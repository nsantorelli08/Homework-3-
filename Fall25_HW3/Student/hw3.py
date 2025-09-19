import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def norm_histogram(histogram):
    """
    takes a list of counts and converts to a list of probabilities, outputs the probability list.
    :param histogram: a numpy ndarray object
    :return: list
    """
    total = sum(histogram)
    norm_hist = [count / total for count in histogram]
    return norm_hist

def compute_j(histogram, bin_width, num_samples):
    """
    takes list of counts, uses norm_histogram function to output the histogram of probabilities,
    then calculates compute_j for one specific bin width (reference: histogram.pdf page19)
    :param histogram: list
    :param bin_width: float
    :param num_samples: int
    :return: float
    """
    sum_sq = sum(p ** 2 for p in histogram)
    j_val = (2 / ((num_samples - 1) * bin_width)) - ((num_samples + 1) / ((num_samples - 1) * bin_width)) * sum_sq
    return j_val

def sweep_n(data, min_val, max_val, min_bins, max_bins):
    """
    find the optimal bin
    calculate compute_j for a full sweep [min_bins to max_bins]
    please make sure max_bins is included in your sweep

    The variable "data" is the raw data that still needs to be "processed"
    with matplotlib.pyplot.hist to output the histogram

    You must utilize the variables (data, min_val, max_val, min_bins, max_bins)
    in your code for 'sweep_n' to determine the correct input to the function 'matplotlib.pyplot.hist',
    specifically the values to (x, bins, range).
    Other input variables of 'matplotlib.pyplot.hist' can be set as default value.

    :param data: list
    :param min_val: int
    :param max_val: int
    :param min_bins: int
    :param max_bins: int
    :return: list
    """
    j_values = []
    num_samples = len(data)

    for n_bins in range(min_bins, max_bins + 1):
        bin_width = (max_val - min_val) / n_bins
        counts, bins, patches = plt.hist(data, bins=n_bins, range=(min_val, max_val))
        probabilities = norm_histogram(counts)
        j_val = compute_j(probabilities, bin_width, num_samples)
        j_values.append(j_val)
        plt.clf()

    return j_values

def find_min(l):
    """
    takes a list of numbers and returns the three smallest number in that list and their index.
    return a dict i.e.
    {index_of_the_smallest_value: the_smallest_value, index_of_the_second_smallest_value: the_second_smallest_value, ...}

    For example:
        A list(l) is [14,27,15,49,23,41,147]
        Then you should return {0: 14, 2: 15, 4: 23}

    :param l: list
    :return: dict: {int: float}
    """
    float_list = []
    for value in l:
        float_list.append(float(value))

    pairs = []

    for i in range(len(float_list)):
        pairs.append([float_list[i], i])

    result = {}

    for i in range(3):
        if not pairs:
            break
        first_value, first_index = pairs[0]
        min_val = first_value
        min_index = first_index
        min_pos = 0
        for j in range(1, len(pairs)):
            current_value, current_index = pairs[j]
            if current_value < min_val:
                min_val = current_value
                min_index = current_index
                min_pos = j
        result[min_index] = min_val
        pairs.pop(min_pos)

    return result

# ============================== P2 ==================================

def get_data(filename):
    return np.loadtxt(filename)

def get_coordinates(data, each_dist):
    # Part B
    """
    calculates the QQ plot given an array of data and a name of a distribution
    outputs a tuple of 2 numpy arrays from the output of the QQ plot
    :param data: np.ndarray
    :param each_dist: str
    :return: (np.ndarray, np.ndarray)
    """
    dist_map = {
        'norm': 'norm',
        'expon': 'expon',
        'uniform': 'uniform',
        'wald': 'wald'
    }
    res = stats.probplot(data, dist=dist_map[each_dist], plot=None)
    return res[0]


def calculate_distance(x, y):
    # Part B
    """
    calculates the projected distance between x and y
    returns the distance as a float
    :param x: float
    :param y: float
    :return: float
    """
    mid = (x + y) / 2
    distance_calc = ((x - mid) ** 2 + (y - mid) ** 2) ** 0.5
    return float(distance_calc)


def find_dist(data):
    # Part B
    """
    from a dictionary of distribution names and their respective errors, finds the distribution having the minimum value
    outputs the minimum value and the name of the distribution
    :param data: dict: {str: float}
    :return: (str, float)
    """
    min_dist = min(data, key=data.get)
    return (min_dist, data[min_dist])

def main(data_file):
    """
    Input a csv file and return distribution type, the error corresponding to the distribution type (e.g. return ('norm', 0.32))
    :param: *.csv file name (str)
    :return: (str, float)
    """
    data = get_data(data_file)
    dists = ("norm", "expon", "uniform", "wald")
    sum_err = [0] * 4
    for ind, each_dist in enumerate(dists):
        X, Y = get_coordinates(data, each_dist)
        for x, y in zip(X, Y):
            sum_err[ind] += calculate_distance(x, y)
    return find_dist(dict(zip(dists, sum_err)))


if __name__ == "__main__":
    data = np.loadtxt("input.txt")  # reads data from input.txt
    lo = min(data)
    hi = max(data)
    bin_l = 1
    bin_h = 100
    js = sweep_n(data, lo, hi, bin_l, bin_h)
    """
    the values bin_l and bin_h represent the lower and higher bound of the range of bins.
    They will change when we test your code and you should be mindful of that.
    """
    print(find_min(js))
  ############### Uncomment for P2 #################

for each_dataset in [
    "sample_norm.csv",
    "sample_expon.csv",
    "sample_uniform.csv",
    "sample_wald.csv",
    "distA.csv",
    "distB.csv",
    "distC.csv",
     ]:
    print(main(each_dataset))
