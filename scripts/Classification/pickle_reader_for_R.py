from ateamopt.utils import utility

def read_pickle_file(file):
    pickle_data = utility.load_pickle(file)
    return pickle_data