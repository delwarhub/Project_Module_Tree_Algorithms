import os
import yaml
from utils import config
from data.biocreative_dataset import Biocreative_dataset
from models import train_RandomForestClassifier, evaluate, load_trained_model
from XAI import ConfusionMatrixDisplay, shap_global_and_local_interpretability, lime_local_interpretability

if __name__ == "__main__":
    
    biocreative_dataset = Biocreative_dataset(path_to_train_file=config["PATH_TO_TRAIN_FILE"], 
                                              path_to_test_file=config["PATH_TO_TEST_FILE"], 
                                              feature_column='abstract')
    # generate feature encodings
    X_train, X_test, vectorizer = biocreative_dataset.generate_features(encoding_type="tfidf", # tfidf or bow
                                                                        stop_words="english")

    y_train, y_test = biocreative_dataset.train_df['label'].to_numpy(), biocreative_dataset.test_df['label'].to_numpy() # get ground-truth labels

    if not config["USE_TRAINED_MODEL"]:

        # train RandomForestClassifier & evaluate
        model = train_RandomForestClassifier(X_train=X_train,
                                            y_train=y_train,
                                            n_estimator=100,
                                            model_name="100_RFC")
        evaluate(model=model, X_test=X_test, y_test=y_test, labels=biocreative_dataset.labels)
    else:
        model = load_trained_model(os.path.join(config["PATH_TO_MODEL_OUTPUT_DIR"], config["PATH_TO_TRAINED_MODEL"]))

        sgli = shap_global_and_local_interpretability(model,
                                                      vectorizer,
                                                      X_train.toarray(),
                                                      X_test.toarray(),
                                                      biocreative_dataset.labels)
        li = lime_local_interpretability()

        if config["PLOT_SHAP_SUMMARY_PLOT"]:
            sgli.overall_summary_plot()

        elif config["PLOT_SHAP_CLASSWISE_SUMMARY_PLOT"]:
            for classname in biocreative_dataset.labels:
                sgli.classwise_summary_plot(classname=classname)

        elif config["PLOT_SHAP_FORCE_PLOT"]:
            for row_id, classname in zip(config["FP_SAMPLE_INDICES"], config["FP_SAMPLE_CLASSES"]):
                sgli.force_plot(classname, row_id)

        elif config["PLOT_SHAP_WATERFALL_PLOT"]:
            for row_id, classname in zip(config["WP_SAMPLE_INDICES"], config["WP_SAMPLE_CLASSES"]):
                sgli.waterfall_plot(classname, row_id)

        elif config["PLOT_LIME_TOP_K"]:
            li.generate_top_k_class_labels(top_k=config["TOP_K"], row_id=config["TOP_K_SAMPLE_INDEX"])

        elif config["PLOT_LIME_WITH_TEXT"]:
            for row_id, classname in zip(config["LI_SAMPLE_INDICES"], config["LI_SAMPLE_CLASSES"]):
                li.explainer.explain_instance(biocreative_dataset.test_df.iloc[row_id]["abstract"], 
                                              li.pipeline.predict_proba, num_features=10, 
                                              labels=biocreative_dataset.labels).as_html(labels=(0,))

