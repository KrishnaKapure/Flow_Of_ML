import pandas as pd
from sklearn.model_selection import train_test_split
#These lines import required libraries. pandas is imported with the alias pd for reading and manipulating datasets. 
# train_test_split is a function from scikit-learn that helps split the data into training and testing sets.


def main():
# This line defines the main function, which contains the core logic to be executed.

    df = pd.read_csv("iris.csv")
    # Reads the Iris dataset from a file named "iris.csv" into a pandas DataFrame called df

    print("Dataset loaded succesfully")
    print(df["variety"])
    #Prints a message to confirm the dataset was loaded. 
    #Then prints the variety column, which contains the labels/categories (species) for each Iris data sample.

    df["variety"] = df["variety"].map({'Setosa' :0, 'Versicolor': 1, 'Virginica': 2})
    # Maps the string classes in the variety column to numeric codes: 'Setosa' becomes 0, 'Versicolor' becomes 1, and 'Virginica' becomes 2.
    # This step is necessary because many machine learning models require numeric input for target variables.

    X = df.drop('variety', axis ='columns')
    Y = df["variety"]
    # Separates the features (X) from the target labels (Y).
    # X is the DataFrame without the variety column, while Y is just the variety column.

    X_train, X_test, Y_train, Y_test = train_test_split (X,Y, test_size=0.2)
    # Randomly splits the dataset into training and testing subsets.
    # 80% is used for training models, and 20% for testing/evaluating accuracy.

    print("Diamention of X_train:",X_train.shape)
    print("Diamention of X_test:",X_test.shape)
    print("Diamention of Y_train:",Y_train.shape)
    print("Diamention of Y_test:",Y_test.shape)

    # Prints out the shape of each resulting subset (number of samples and features for X, number of samples for Y) to ensure the split was done correctly.


if __name__ =="__main__":
    main()
# Ensures that the main() function is only run when the script is executed directly, not when it is imported as a module in another script.
