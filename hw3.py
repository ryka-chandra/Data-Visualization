'''
Ryka Chandra
Intermediate Data Programming
This program implements the functions assigned in HW3 Write-up
'''

import os
import tempfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()


def compare_bachelors_1980(df):
    """
    Takes the data and computes the percentages of men and women
    who achieved a minimum degree of a Bachelor’s degree in 1980
    in a 2-by-2 DataFrame with rows corresponding to men and women
    and columns corresponding to Sex and Total
    """
    df2 = df[(df['Year'] == 1980) & (df['Min degree'] ==
                                     'bachelor\'s') & (df['Sex'] != 'A')]
    return df2.loc[:, ['Sex', 'Total']]


def top_2_2000s(data, sex='A'):
    """
    Takes two arguments, the data and a sex parameter, and computes the
    two most commonly earned degrees for that given sex between the
    years 2000 and 2010 (inclusive)
    """
    df = data[(data['Sex'] == sex) & (data['Year'].between(2000, 2010))]
    df2 = df.groupby(['Min degree'])['Total'].mean()
    return df2.sort_values(ascending=False).head(2)


def line_plot_bachelors(data):
    """
    Takes the data and plots a line chart of the total percentages of all
    people Sex A with bachelor's Min degree over time
    """
    df = data[(data['Sex'] == 'A') & (data['Min degree'] == 'bachelor\'s')]
    df2 = df.groupby(['Year'])['Total'].mean().reset_index()
    sns.relplot(x='Year', y='Total', kind='line', data=df2)
    plt.title('Percentage Earning Bachelor’s over Time')
    plt.ylabel('Percentage')
    plt.savefig('line_plot_bachelors.png', bbox_inches='tight')


def bar_chart_high_school(data):
    """
    Takes the data and plots a bar chart comparing the total percentages of Sex
    F, M, and A with high school Min degree in the Year 2009
    """
    df = data[(data['Year'] == 2009) & (data['Min degree'] == 'high school')]
    sns.catplot(data=df, kind='bar', x='Sex', y='Total')
    plt.title('Percentage Completed High School by Sex')
    plt.ylabel('Percentage')
    plt.savefig('bar_chart_high_school.png', bbox_inches='tight')


def plot_hispanic_min_degree(data):
    """
    Takes the data and plots how the percentage of Hispanic people
    with degrees have changed between 1990–2010 (inclusive) for
    high school and bachelor's Min degree
    """
    df = data[((data['Min degree'] == 'bachelor\'s') |
               (data['Min degree'] ==
                'high school')) & (data['Year'].between(1990, 2010))]
    df2 = df.groupby(['Year', 'Min degree'])['Hispanic'].mean().reset_index()
    sns.relplot(x='Year', y='Hispanic', hue='Min degree',
                kind='line', data=df2)
    plt.title(
        'Percentage of Hispanics Earning Bachelor’s vs High School Over Time')
    plt.ylabel('Percentage')
    plt.savefig('plot_hispanic_min_degree.png', bbox_inches='tight')


def fit_and_predict_degrees(data):
    """
    Takes the data and returns the test mean squared error as a float
    """
    data = data.iloc[:, 0:4]
    data = data.dropna()
    features = data.loc[:, data.columns != 'Total']
    labels = data['Total']
    features = pd.get_dummies(features)
    f_train, f_test, l_train, l_test = train_test_split(
        features, labels, train_size=0.8)
    model = DecisionTreeRegressor()
    model.fit(f_train, l_train)
    label_predictions = model.predict(f_test)
    test_accuracy = mean_squared_error(l_test, label_predictions)
    return test_accuracy


def main():
    """
    Calls the main method, loads in the dataset provided,
    and calls all of the functions
    """
    data = pd.read_csv('nces-ed-attainment.csv', na_values=['---'])
    sns.set()
    compare_bachelors_1980(data)
    top_2_2000s(data, sex='A')
    line_plot_bachelors(data)
    bar_chart_high_school(data)
    plot_hispanic_min_degree(data)
    fit_and_predict_degrees(data)


if __name__ == '__main__':
    main()
