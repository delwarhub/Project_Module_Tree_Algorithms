import pandas as pd
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class Biocreative_dataset:
    def __init__(self, path_to_train_file: str,
                 path_to_test_file: str,
                 feature_column: str='abstract'):
        
        self.path_to_train_file = path_to_train_file
        self.path_to_test_file = path_to_test_file
        self.feature_column = feature_column

        self.train_df, self.test_df, self.class_mappings = self.preprocess_data()
        self.labels = list(self.class_mappings.keys())

    def preprocess_data(self):
        """process LitCovid dataset & transform to multiclass setting"""

        train_df = pd.read_csv(self.path_to_train_file)
        test_df = pd.read_csv(self.path_to_test_file)
        
        # drop all columns except `abstract`, and `label`
        train_df.drop([label for label in train_df.columns.tolist() if label not in [self.feature_column, 'label']], axis=1, inplace=True)
        test_df.drop([label for label in test_df.columns.tolist() if label not in [self.feature_column, 'label']], axis=1, inplace=True)

        # get unique set of labels using `train_df`
        unique_label_set = train_df['label'].apply(lambda row: row.split(';')).tolist()
        labels = list(set(itertools.chain(*unique_label_set)))
        
        # select multiclass instances 
        train_df = train_df[train_df['label'].apply(lambda row: len(row.split(';')) == 1)]
        test_df = test_df[test_df['label'].apply(lambda row: len(row.split(';')) == 1)]

        class_mappings = dict(zip(labels, range(len(labels))))
        train_df['label'] = train_df['label'].map(class_mappings)
        test_df['label'] = test_df['label'].map(class_mappings)

        # drop NaN values
        train_df.dropna(inplace=True)
        test_df.dropna(inplace=True)

        if self.feature_column == 'keywords':
            train_df['keywords'] = train_df.apply(lambda row: " ".join(row['keywords'].split(';')), axis=1)
            test_df['keywords'] = test_df.apply(lambda row: " ".join(row['keywords'].split(';')), axis=1)
            train_df.dropna(inplace=True)
            test_df.dropna(inplace=True)

        return train_df, test_df, class_mappings

    def generate_features(self, 
                          encoding_type='tfidf',
                          stop_words='english'):
        """apply either BOW or TfidF vectorization to input-data"""

        if encoding_type == "tfidf":
            vectorizer = TfidfVectorizer(lowercase=True, 
                                        stop_words=stop_words,
                                        max_features=10000,
                                        norm='l2')
            X_train = vectorizer.fit_transform(self.train_df[self.feature_column])
            X_test = vectorizer.transform(self.test_df[self.feature_column])
        elif encoding_type == "bow":
            vectorizer = CountVectorizer(lowercase=True,
                                        stop_words=stop_words,
                                        max_features=10000)
            X_train = vectorizer.fit_transform(self.train_df[self.feature_column])
            X_test = vectorizer.transform(self.test_df[self.feature_column])
        return X_train, X_test, vectorizer