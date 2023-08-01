import pandas as pd
import numpy as np
import gc

import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.metrics import ConfusionMatrixDisplay

import shap
from lime.lime_text import LimeTextExplainer

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
plt.rcParams['figure.dpi'] = 300
rcParams['figure.figsize'] = 12, 8

def plot_confusion_plot(model, X_test, y_test,
                        labels):
    
    # confusion plot 
    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=labels,
        cmap=plt.cm.Blues,
        xticks_rotation='vertical'
    )
    disp.ax_.set_title("Confusion Plot `abstract`")

    plt.show()

# shap interpretability tools
class shap_global_and_local_interpretability:
    def __init__(self, model: sklearn.ensemble._forest.RandomForestClassifier,
                       vectorizer: sklearn.feature_extraction.text.TfidfVectorizer,
                       X_train: np.ndarray,
                       X_test: np.ndarray,
                       labels: list):
        super(shap_global_and_local_interpretability, self).__init__()

        gc.collect()
        self.model = model
        self.vectorizer = vectorizer
        self.X_train = X_train
        self.X_test = X_test
        self.labels = labels
        self.class_mappings = dict(zip(labels, range(len(labels))))

        self.explainer = shap.TreeExplainer(model)
        self.shap_values = self.explainer.shap_values(X_test, approximate=True)

    def overall_summary_plot(self):

        fig = shap.summary_plot(self.shap_values, self.X_test, 
                                plot_type="bar", max_display=10,
                                feature_names=self.vectorizer.get_feature_names_out().tolist(),
                                class_names=self.labels)
        gc.collect()
        return fig

    def classwise_summary_plot(self, classname: str, max_display=10):

        assert [classname] not in self.labels, "Provide appropriate classname; \
        should be one of the following: {}".format(self.labels)
        
        class_idx = self.class_mappings[classname]
        fig = shap.summary_plot(self.shap_values[class_idx], 
                                self.X_test, max_display=max_display, class_names=self.labels,
                                feature_names=self.vectorizer.get_feature_names_out().tolist())
        
        gc.collect()
        return fig

    def force_plot(self, classname: str, row_id: int, show: bool=True,
                   matplotlib: bool=False):

        assert [classname] not in self.labels, "Provide appropriate classname; \
        should be one of the following: {}".format(self.labels)

        shap.initjs()

        class_idx = self.class_mappings[classname]
        fig = shap.force_plot(self.explainer.expected_value[class_idx],
                              self.shap_values[class_idx][row_id],
                              self.X_test[row_id],
                              feature_names=self.vectorizer.get_feature_names_out().tolist(),
                              matplotlib=matplotlib,
                              show=show)
        
        gc.collect()
        return fig

    def waterfall_plot(self, classname: str, row_id: int):

        assert [classname] not in self.labels, "Provide appropriate classname; \
        should be one of the following: {}".format(self.labels)

        class_idx = self.class_mappings[classname]
        fig = shap.waterfall_plot(shap.Explanation(values=self.shap_values[class_idx][row_id],
                                             base_values=self.explainer.expected_value[class_idx],
                                             data=self.X_test[row_id],
                                             feature_names=self.vectorizer.get_feature_names_out().tolist()))
        
        gc.collect()
        return fig
    
# lime interpretability tools
class lime_local_interpretability:
    def __init__(self, model: sklearn.ensemble._forest.RandomForestClassifier,
                       vectorizer: sklearn.feature_extraction.text.TfidfVectorizer,
                       train_data: pd.DataFrame,
                       test_data: pd.DataFrame,
                       labels: list,
                       text_column: str="abstract",): 
        super(lime_local_interpretability, self).__init__()

        gc.collect()
        self.model = model
        self.vectorizer = vectorizer
        self.train_data = train_data
        self.test_data = test_data
        self.labels = labels
        self.text_column = text_column

        self.class_mappings = dict(zip(labels, range(len(labels))))
        self.pipeline = make_pipeline(vectorizer, model)

        self.explainer = LimeTextExplainer(class_names=labels)
    
    def generate_top_k_class_labels(self, row_id: int, top_k: int=2, 
                                          num_features: int=5):

        exp = self.explainer.explain_instance(self.test_data.iloc[row_id][self.text_column], 
                                         self.pipeline.predict_proba, 
                                         num_features=num_features, 
                                         top_labels=top_k)
        # print(exp.available_labels())

        gc.collect()
        return exp.show_in_notebook(text=False)  

    def  generate_class_labels_w_text(self, row_id: int, classname: str,
                                      num_features: int=5):
        
        assert [classname] not in self.labels, "Provide appropriate classname; \
        should be one of the following: {}".format(self.labels)

        class_idx = self.class_mappings[classname]
        exp = self.explainer.explain_instance(self.test_data.iloc[row_id][self.text_column], 
                                         self.pipeline.predict_proba, 
                                         num_features=num_features,
                                         labels=list(self.class_mappings.values()))
        
        gc.collect()
        return exp.show_in_notebook(text=self.test_data.iloc[row_id][self.text_column], 
                                    labels=(class_idx,))