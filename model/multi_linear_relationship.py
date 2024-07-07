import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

def linear_regression_analysis(X, y):
    # Add a constant term for the intercept
    X = sm.add_constant(X)
    
    # Fit the model
    model = sm.OLS(y, X).fit()
    
    # Return the model summary
    return model

def plot_relationship(X, y, x_label, y_label, title, output_file):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X, y=y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(output_file)
    plt.show()

def main():
    # Load the data
    df = pd.read_csv('./../data/BeijingHousingPrices_processed_modified.csv')

    # Clean and preprocess the data
    df.replace('Unknown', np.nan, inplace=True)
    df.replace('Nan', np.nan, inplace=True)

    df['subway'].fillna(df['subway'].mode()[0], inplace=True)
    
    # Convert communityAverage to numeric, coercing errors
    df['communityAverage'] = pd.to_numeric(df['communityAverage'], errors='coerce')
    df['communityAverage'].fillna(df['communityAverage'].mean(), inplace=True)
    
    df['price'].fillna(df['price'].mean(), inplace=True)

    # Linear regression analysis for subway and price
    X_subway = df[['subway']]
    y_price = df['price']
    model_subway = linear_regression_analysis(X_subway, y_price)
    print("Subway vs Price Regression Summary:")
    print(model_subway.summary())

    # Linear regression analysis for communityAverage and price
    X_community = df[['communityAverage']]
    model_community = linear_regression_analysis(X_community, y_price)
    print("CommunityAverage vs Price Regression Summary:")
    print(model_community.summary())

    # Plot relationships
    plot_relationship(df['subway'], df['price'], 'Subway', 'Price', 'Subway vs Price', './../result/subway_vs_price.png')
    plot_relationship(df['communityAverage'], df['price'], 'Community Average', 'Price', 'Community Average vs Price', './../result/community_average_vs_price.png')

if __name__ == "__main__":
    main()
