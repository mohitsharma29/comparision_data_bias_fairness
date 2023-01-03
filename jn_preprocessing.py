# Preprocessing with Jiang_Nachum Label Bias settings: https://github.com/TengLin1/google-research/blob/b3fb9d003097c786c79b23f6b7bd53ea2b166fc0/label_bias/Label_Bias_EqualOdds.ipynb
import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset

def preprocessAdult(train_path, test_path):
    CATEGORICAL_COLUMNS = [
    'workclass', 'education', 'marital-status', 'occupation', 'relationship',
    'race', 'sex', 'native-country'
    ]
    CONTINUOUS_COLUMNS = [
        'age', 'capital-gain', 'capital-loss', 'hours-per-week'
    ]
    COLUMNS = [
        'age', 'workclass', 'fnlwgt', 'education',
        'marital-status', 'occupation', 'relationship', 'race', 'gender',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
        'income'
    ]
    LABEL_COLUMN = 'income'

    PROTECTED_GROUPS = [
        'sex_Female', 'sex_Male', 'race_White', 'race_Non-white'
    ]
    def get_adult_data():
      train_file = train_path
      test_file = test_path

      train_df_raw = pd.read_csv(train_file)
      test_df_raw = pd.read_csv(test_file)
      train_df_raw[LABEL_COLUMN] = (
          train_df_raw['income'].apply(lambda x: '>50K' in x)).astype(int)
      test_df_raw[LABEL_COLUMN] = (
          test_df_raw['income'].apply(lambda x: '>50K' in x)).astype(int)
      # Preprocessing Features
      pd.options.mode.chained_assignment = None  # default='warn'

      # Functions for preprocessing categorical and continuous columns.
      def binarize_categorical_columns(input_train_df,
                                       input_test_df,
                                       categorical_columns=[]):

        def fix_columns(input_train_df, input_test_df):
          test_df_missing_cols = set(input_train_df.columns) - set(
              input_test_df.columns)
          for c in test_df_missing_cols:
            input_test_df[c] = 0
          train_df_missing_cols = set(input_test_df.columns) - set(
              input_train_df.columns)
          for c in train_df_missing_cols:
            input_train_df[c] = 0
          input_train_df = input_train_df[input_test_df.columns]
          return input_train_df, input_test_df

        # Binarize categorical columns.
        binarized_train_df = pd.get_dummies(
            input_train_df, columns=categorical_columns)
        binarized_test_df = pd.get_dummies(
            input_test_df, columns=categorical_columns)
        # Make sure the train and test dataframes have the same binarized columns.
        fixed_train_df, fixed_test_df = fix_columns(binarized_train_df,
                                                    binarized_test_df)
        return fixed_train_df, fixed_test_df

      def bucketize_continuous_column(input_train_df,
                                      input_test_df,
                                      continuous_column_name,
                                      num_quantiles=None,
                                      bins=None):
        assert (num_quantiles is None or bins is None)
        if num_quantiles is not None:
          train_quantized, bins_quantized = pd.qcut(
              input_train_df[continuous_column_name],
              num_quantiles,
              retbins=True,
              labels=False)
          input_train_df[continuous_column_name] = pd.cut(
              input_train_df[continuous_column_name], bins_quantized, labels=False)
          input_test_df[continuous_column_name] = pd.cut(
              input_test_df[continuous_column_name], bins_quantized, labels=False)
        elif bins is not None:
          input_train_df[continuous_column_name] = pd.cut(
              input_train_df[continuous_column_name], bins, labels=False)
          input_test_df[continuous_column_name] = pd.cut(
              input_test_df[continuous_column_name], bins, labels=False)

      # Filter out all columns except the ones specified.
      train_df = train_df_raw
      test_df = test_df_raw
      # Bucketize continuous columns.
      bucketize_continuous_column(train_df, test_df, 'age', num_quantiles=4)
      bucketize_continuous_column(
          train_df, test_df, 'capital-gain', bins=[-1, 1, 4000, 10000, 100000])
      bucketize_continuous_column(
          train_df, test_df, 'capital-loss', bins=[-1, 1, 1800, 1950, 4500])
      bucketize_continuous_column(
          train_df, test_df, 'hours-per-week', bins=[0, 39, 41, 50, 100])
      train_df, test_df = binarize_categorical_columns(
          train_df,
          test_df,
          categorical_columns=CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS)
      feature_names = list(train_df.keys())
      feature_names.remove(LABEL_COLUMN)
      num_features = len(feature_names)
      return train_df, test_df, feature_names


    train_df, test_df, feature_names = get_adult_data()
    #X_train_adult = np.array(train_df[feature_names])
    #y_train_adult = np.array(train_df[LABEL_COLUMN])
    #X_test_adult = np.array(test_df[feature_names])
    #y_test_adult = np.array(test_df[LABEL_COLUMN])
    
    #protected_train_adult = [np.array(train_df[g]) for g in PROTECTED_GROUPS]
    #protected_test_adult = [np.array(test_df[g]) for g in PROTECTED_GROUPS]
    #y_train = train_df[LABEL_COLUMN]
    #y_test = train_df[LABEL_COLUMN]
    #z_train = train_df['sex_Male']
    #z_train = z_train.reset_index(name='sex').drop(columns=['index'])
    #z_test = test_df['sex_Male']
    #z_test = z_test.reset_index(name='sex').drop(columns=['index'])
    train_df = train_df.drop(columns=['race_White', 'race_Non-white', 'sex_Female'])
    test_df = test_df.drop(columns=['race_White', 'race_Non-white', 'sex_Female'])
    train_df = train_df.rename({'sex_Male': 'sex'}, axis=1)
    test_df = test_df.rename({'sex_Male': 'sex'}, axis=1)
    
    # Convert to AIF format
    binary_train_dataset = BinaryLabelDataset(favorable_label=1,
                                unfavorable_label=0,
                                df=train_df,
                                label_names=['income'],
                                protected_attribute_names=['sex'],
                                unprivileged_protected_attributes=[{'sex': 0}])
    binary_test_dataset = BinaryLabelDataset(favorable_label=1,
                                unfavorable_label=0,
                                df=test_df,
                                label_names=['income'],
                                protected_attribute_names=['sex'],
                                unprivileged_protected_attributes=[{'sex': 0}])
    return binary_train_dataset, binary_test_dataset