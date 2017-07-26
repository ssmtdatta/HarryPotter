import pickle


def pickleSomething(data, path, pickle_filename):
    with open(path+pickle_filename, 'wb') as p:
        pickled = pickle.dump(data, p)
    return pickled

def unpickleSomething(path, pickle_filename):
    with open(path+pickle_filename, 'rb') as p: 
        unpickled = pickle.load(p)
        return unpickled

