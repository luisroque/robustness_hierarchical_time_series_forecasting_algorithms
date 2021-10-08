
def create_new_version(y, transformations):
    y_new = np.zeros((y.shape[0], y.shape[1]))

    for i in range(dataset.number_series):
        y_new[i] =  manipulate_data(x, np.random.choice(transformations)).apply_transf()
    
    return y_new