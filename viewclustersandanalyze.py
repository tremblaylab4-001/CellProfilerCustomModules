import os
import io
import urllib
import csv
import subprocess
import platform
from pathlib import Path
import cellprofiler_core.module as cpm
import cellprofiler_core.setting as cps
from cellprofiler_core.workspace import Workspace
from cellprofiler_core.setting.text import Directory
from cellprofiler_core.setting.text import Filename
from cellprofiler_core.constants.module import IO_FOLDER_CHOICE_HELP_TEXT
#from cellprofiler_core.preferences import NO_FOLDER_NAME
from cellprofiler_core.preferences import URL_FOLDER_NAME
from cellprofiler_core.preferences import get_data_file
from cellprofiler_core.preferences import is_url_path
from cellprofiler_core.preferences import get_default_output_directory
from cellprofiler_core.preferences import get_default_image_directory
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting import ValidationError
import cellprofiler_core.setting.text as cps_text
#from cellprofiler_core.setting.do_something import DoSomething
#from cellprofiler_core.setting.multichoice import MultiChoice
#from cellprofiler_core.setting.range import IntegerRange
from cellprofiler_core.utilities.image import generate_presigned_url
from cellprofiler_core.utilities.core.modules.load_data import header_to_column

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import wx
import time

"""Cache of header columns for files"""
header_cache = {}

class ViewClustersAndAnalyze(cpm.Module):
    # 1. Basic Metadata
    module_name = "ViewClustersAndAnalyze"
    variable_revision_number = 1
    category = "Utility"

    def create_settings(self):
        # 2. Define user-configurable settings
        self.text_message = cps.text.Text(
            "ViewClustersAndAnalyze", "ViewClustersAndAnalyze",
            doc="ViewClustersAndAnalyze"
        )
        self.open_csv_upon_complete = Binary(
            "Open labelled output spreadsheet after module is finished?",
            False,
            doc="""\
Select *{YES}* to open the produced spreadsheet using your system's default spreadsheet viewer.
""".format(
                **{"YES": "Yes"}
            ),
        )

    def settings(self):
        # Return a list of all settings
        return [self.open_csv_upon_complete]
        
    def visible_settings(self):
        # Return settings visible in the GUI
        return [self.open_csv_upon_complete]
    
    def validate_module(self, pipeline):
        return [self.open_csv_upon_complete]
    
    def run(self, workspace):
        print("==== RUNNING ====")

        # Load CSVs from Module2
        labels_cleaned_path = workspace.measurements.get_current_image_measurement("OutlierAndClusterSettings_labels_cleaned_path")
        labels_cleaned = np.load(labels_cleaned_path,allow_pickle=True)

        X_pca_df_path = workspace.measurements.get_current_image_measurement("OutlierAndClusterSettings_X_pca_df_path")
        X_pca_df = pd.read_csv(X_pca_df_path)

        df_path = workspace.measurements.get_current_image_measurement("OutlierAndClusterSettings_df_path")
        df = pd.read_csv(df_path)

        #X_scaled_path = workspace.measurements.get_current_image_measurement("OutlierAndClusterSettings_X_scaled_path")
        #X_scaled = np.load(X_scaled_path,allow_pickle=True)
        # ^ Not Needed?
  
        X_pd_path = workspace.measurements.get_current_image_measurement("OutlierAndClusterSettings_X_pd_path")
        X_pd = pd.read_csv(X_pd_path)

        linked_path = workspace.measurements.get_current_image_measurement("OutlierAndClusterSettings_linked_path")
        linked = np.load(linked_path)

        num_clusters = workspace.measurements.get_current_image_measurement("OutlierAndClusterSettings_num_clusters")
        num_clusters = int(num_clusters)
        
        scaler = StandardScaler()

        distance_threshold = num_clusters
        hierarchical_labels = fcluster(linked, t=distance_threshold, criterion='distance')
        n_clusters = len(np.unique(hierarchical_labels))

        kmeans = KMeans(n_clusters = n_clusters, init="k-means++", random_state=24) # Ward distance specified by user at end of module 2
        # X_scaled = scaler.fit_transform(X_pd)
        # ^ Not Needed?
        X_pd["Cluster"] = kmeans.fit_predict(X_pd)

        # Cluster labels for each point
        X_pd["cluster_labels"] = kmeans.labels_ + 1

        # create figure of clustering assignment
        fig = go.Figure()
        color_map = {
            1: "red",
            2: "blue",
            3: "green",
            4: "yellow",
            5: "purple",
            6: "pink",
            7: "aquamarine",
            8: "violet",
            9: "royalblue",
            10: "brown",
            11: "darkgreen"
        }
        X_pd["PC1"] = X_pd.iloc[:,0]
        X_pd["PC2"] = X_pd.iloc[:,1]
        X_pd["PC3"] = X_pd.iloc[:,2]

        labels_cleaned = pd.DataFrame(data = labels_cleaned, columns = ["ObjectNumber", "FileName_Blind", "Condition"])
        labels_new = pd.concat([X_pd["cluster_labels"], labels_cleaned], axis=1)
        labels_new.to_csv((get_default_image_directory() + "/" + "labels_new_5.csv"))

        colors = X_pd["cluster_labels"].map(color_map)
        scatter = go.Scatter3d(x = X_pd["PC1"], y = X_pd["PC2"], z = X_pd["PC3"], mode = "markers", marker = dict(size = 8, color = colors),
                        customdata=labels_new[["Condition", "FileName_Blind", "ObjectNumber", "cluster_labels"]].values,
                        hovertemplate=("Condition = %{customdata[0]}<br>" +
                                      "image = %{customdata[1]}<br>" +
                                      "object = %{customdata[2]}<br>" +
                                      "cluster = %{customdata[3]}"))
        fig.add_trace(scatter)
        fig.layout.title.text = "Clusters"
        dir(go)
        fig.show()

        df_new = pd.DataFrame({"clusters": X_pd["cluster_labels"],
        "groups": labels_cleaned["Condition"]
        })

        table = pd.crosstab(df_new["clusters"], df_new["groups"])
        print(table)

        table_long = table.stack().reset_index(name="Freq")
        print(table_long)

        # creating barplot
        sns.barplot(x = "groups", y = "Freq", hue = "clusters", data = table_long, palette = color_map)
        plt.show()
        # end of module 3
        # barplot and table output of module 3
        labeled_output= pd.concat([labels_cleaned, X_pd["cluster_labels"]], axis=1)
        labeled_output_filepath = (get_default_image_directory() + "/" + "labeled_output.csv")
        labeled_output.to_csv(labeled_output_filepath)


        if self.open_csv_upon_complete:
            if platform.system() == 'Linux':
                subprocess.call(['xdg-open', labeled_output_filepath])
            elif platform.system() == 'darwin':
                subprocess.call(['open', labeled_output_filepath])
        #if self.open_csv_upon_complete:
        #    if


        #print(n_components)