# Importing libraries
 
import numpy as np
 
import pandas as pd
 
from sklearn.model_selection import train_test_split
 
import matplotlib.pyplot as plt
 
from sklearn.linear_model import LinearRegression
 
# driver code
 
def main() :
     
    # Create dataset
     
    X = np.array( [ [1], [2], [3], [4], [5], [6], [7] ] )
     
    Y = np.array( [ 45000, 50000, 60000, 80000, 110000, 150000, 200000 ] )
     
    # Model training
     
    model = LinearRegression()
 
    model.fit( X, Y )
     
    # Prediction
 
    Y_pred = model.predict( X )
     
    # Visualization
     
    plt.scatter( X, Y, color = 'blue' )
     
    plt.plot( X, Y_pred, color = 'orange' )
     
    plt.title( 'X vs Y' )
     
    plt.xlabel( 'X' )
     
    plt.ylabel( 'Y' )
     
    plt.show()
     
     
if __name__ == "__main__" :
     
    main()
