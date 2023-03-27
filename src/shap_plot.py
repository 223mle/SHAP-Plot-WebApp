import shap
import pandas as pd
import matplotlib.pyplot as plt


class ShapPlot:
    def __init__(self, model, train):
        self.model = model
        self.train = train
        self.shap_values = shap.TreeExplainer(model, data=train).shap_values(train)

    def summary_plot(self):
        explainer = shap.TreeExplainer(self.model, data=self.train)
        shap_val = explainer.shap_values(self.train)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values=shap_val,
                        features=self.train,
                        feature_names=self.train.columns)
        return fig

    def summary_plot_bar(self):
        explainer = shap.TreeExplainer(self.model, data=self.train)
        shap_val = explainer.shap_values(self.train)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values=shap_val,
                        features=self.train,
                        feature_names=self.train.columns,
                        plot_type='bar')
        return fig

    def waterfall_plot(self, num: int):
        explainer = shap.TreeExplainer(self.model, data=self.train)
        shap_vals = explainer(self.train)
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_vals[num])
        return fig



    #def dependence_plot(self, column: str):
    #    shap_values = self.shap_values
    #    train = self.train
    #    fig, ax = plt.subplots()
    #    shap.dependence_plot(column, shap_values, train)
    #    return fig
