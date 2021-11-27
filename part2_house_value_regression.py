import torch
from torch import nn
import pickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.metrics import max_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split

OCEAN_PROXIMITY_LABELS = [
                        "NEAR BAY",
                        "INLAND",
                        "NEAR OCEAN",
                        "<1H OCEAN",
                        "ISLAND",
                    ]


class Regressor(BaseEstimator):

    def __init__(self, x, nb_epoch = 1000, learning_rate=0.001, batch_size=4096, layers=[100, 50, 25, 12, 1]):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own

        self.lb = LabelBinarizer()
        self.lb.fit(OCEAN_PROXIMITY_LABELS)

        self.scaler = StandardScaler()

        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch 
        self.layers = layers


        nn_layers = []
        prev_layer = self.input_size
        for i, layer in enumerate(layers):
            nn_layers.append(nn.Linear(prev_layer, layer))
            if i != len(layers) - 1:
                nn_layers.append(nn.ReLU()) 
            prev_layer = layer

        self.model = torch.nn.Sequential(*nn_layers)
        # print(self.model)
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size).
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # We fill in missing values
        # Only 'total_bedrooms' column has NaN values - we know this from x.isna().sum()
        
        TOTAL_BEDROOMS_DEFAULT_VALUE = 0.0
        values = {"total_bedrooms": TOTAL_BEDROOMS_DEFAULT_VALUE}

        x_filled = x.fillna(value=values)

        # We one-hot encode textual values
        encodings = self.lb.transform(x_filled['ocean_proximity'])
        encodings_df = pd.DataFrame(encodings, columns=self.lb.classes_)
        x_ocean_dropped = x_filled.drop(columns=['ocean_proximity'])

        x_ocean_dropped.reset_index(drop=True, inplace=True)
        encodings_df.reset_index(drop=True, inplace=True)
        x_encoded = pd.concat([x_ocean_dropped, encodings_df], axis=1)
        # print(x_encoded)

        # We normalize each feature
        if training:
            self.scaler.fit(x_encoded.values)
        
        x_fitted = self.scaler.transform(x_encoded.values)
        x_scaled = pd.DataFrame(x_fitted, columns=x_encoded.columns)

        assert x_encoded.isna().sum().sum() == 0
        assert x_encoded.shape[1] == 13
        assert x_scaled.min().all() >= 0
        assert x_scaled.max().all() <= 1

        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None
        return torch.tensor(x_scaled.values).float(), torch.tensor(y.values).float() if y is not None else None

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
        
        self.model.train()
        
        for epoch in range(self.nb_epoch):

            for batch_idx in range(0, X.size(dim=0), self.batch_size):
                output = self.model(X[batch_idx : batch_idx + self.batch_size])
                
                self.optimizer.zero_grad()
                loss = self.loss_func(output, Y[batch_idx : batch_idx + self.batch_size])
                loss.backward()

                self.optimizer.step()
                # print("Epoch {}: Loss {}".format(epoch + 1, loss.item()))

        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget

        self.model.eval()

        output = self.model(X)
        return output.detach().numpy()

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        predicted = self.predict(x)
        mse = mean_squared_error(y, predicted, squared=False)
        return mse 

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(x, y):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
        - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    param_grid = [{
        'layers': [[100, 50, 25, 1], [100, 50, 25, 12, 1], [200, 100, 50, 25, 1], [25, 12, 6, 1]],
        'batch_size': [512, 1024, 4096],
        'nb_epoch': [1000, 2000, 3000]
    }]

    clf = GridSearchCV(RegressorAdaptor(), param_grid, n_jobs=-1, verbose=3)
    clf.fit(x, y)
    print(clf.cv_results_)
    print(clf.best_params_)
    print("Best score:")
    print(clf.best_score_)

    return clf.best_params_, clf.best_score_ # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


class RegressorAdaptor(BaseEstimator):
    def __init__(
        self,
        nb_epoch=1000,
        learning_rate=0.001,
        layers=[100, 50, 25, 1],
        batch_size=64,
    ):
        self.nb_epoch = nb_epoch
        self.layers = layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def fit(self, X, y):
        self.regressor = Regressor(x=X, **self.get_params())
        self.regressor.fit(X, y)
        return self

    def predict(self, X):
        return self.regressor.predict(X)

    def score(self, X, y=None):
        return -self.regressor.score(X, y)


def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv") 

    # Spliting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    # RegressorHyperParameterSearch(x_train, y_train)

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 1000)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    # regressor = load_regressor()
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))



if __name__ == "__main__":
    example_main()

