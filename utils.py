import yaml
import pickle

# utitlity functions
def save_to_pickle(path_to_file, itemlist):
    """ extenson -> .pkl """
    with open(path_to_file, 'wb') as fp:
        pickle.dump(itemlist, fp)
    print(f"saved file as pickle @ location: {path_to_file}")

def load_from_pickle(path_to_file):
    with open (path_to_file, 'rb') as fp:
        itemlist = pickle.load(fp)
    return itemlist

def load_config(path_to_file):
    with open(path_to_file, 'r') as stream:
        config = yaml.safe_load(stream)
    return config

config = load_config("./config.yaml")
