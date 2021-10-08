
class create_new_version_dataset():
    """
    Create a new version of a dataset. It works by performing one
    one transformation randomly chosen from a list of possible
    transformations and apply it to a random time series. This process
    is done for every time series in the dataset.

    Each time series is modified by one transformation only,
    no cumulative transformations are applied
    """

    def __init__(y, transformations):
        self.y = y
        self.transformations = transformations

    def apply_transformations(self):
        y_new = np.zeros((self.y.shape[0], self.y.shape[1]))

        for i in range(dataset.number_series):
            y_new[i] =  manipulate_data(self.y[i], np.random.choice(self.transformations)).apply_transf()
        
        return y_new