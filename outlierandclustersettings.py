import os
import io
import urllib
import csv

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
from cellprofiler_core.setting.text import Directory, Text
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
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import wx as wx

"""Cache of header columns for files"""
header_cache = {}

class OutlierAndClusterSettings(cpm.Module):
    # 1. Basic Metadata
    module_name = "OutlierAndClusterSettings"
    variable_revision_number = 1
    category = "Utility"

    def create_settings(self):
        # 2. Define user-configurable settings
        self.pick_outliers = Binary(
            "Provide outliers to remove",
            False,
            doc="""\
Select *{YES}* to provide a field that allows you to list which
objects are considered outliers.
""".format(
                **{"YES": "Yes"}
            ),
        )

        self.outliers_list = Text(
                    "outliers_list",
                    "0",
                    doc="""\
        (*Used only if “Provide outliers to remove” is "Yes"*)
        
        Supply a comma-separated list of integers (e.g. 18,40,299) to identify the outliers.
                    """
                    % globals(),
        )

        self.csv_directory = Directory(
            "Input data file location",
            allow_metadata=False,
            support_urls=False,
            doc="""\
Select the folder containing the CSV file to be loaded. {IO_FOLDER_CHOICE_HELP_TEXT}
""".format(
                **{"IO_FOLDER_CHOICE_HELP_TEXT": IO_FOLDER_CHOICE_HELP_TEXT}
            )
        )

        def get_directory_fn():
            """Get the directory for the CSV file name"""
            return self.csv_directory.get_absolute_path()

        def set_directory_fn(path):
            dir_choice, custom_path = self.csv_directory.get_parts_from_path(path)
            self.csv_directory.join_parts(dir_choice, custom_path)

        self.csv_file_name = Filename(
            "Name of the file",
            "None",
            doc="""Provide the file name of the CSV file containing the data you want to load.""",
            get_directory_fn=get_directory_fn,
            set_directory_fn=set_directory_fn,
            browse_msg="Choose CSV file",
            exts=[("Data file (*.csv)", "*.csv"), ("All files (*.*)", "*.*")],
        )

    def settings(self):
        # Return a list of all settings
        return [self.pick_outliers, self.outliers_list, self.csv_directory, self.csv_file_name]

    def visible_settings(self):
        # Return settings visible in the GUI
        results = [self.pick_outliers]
        if self.pick_outliers:
            results += [self.outliers_list]
        return results

    @property
    def csv_path(self):
        """The path and file name of the CSV file to be loaded"""
        if get_data_file() is not None:
            return get_data_file()
        if self.csv_directory.dir_choice == URL_FOLDER_NAME:
            return self.csv_file_name.value

        path = self.csv_directory.get_absolute_path()
        return os.path.join(path, self.csv_file_name.value)

    def open_csv(self, do_not_cache=False):
        """Open the csv file or URL, returning a file descriptor"""
        global header_cache

        if is_url_path(self.csv_path):
            if self.csv_path not in header_cache:
                header_cache[self.csv_path] = {}
            entry = header_cache[self.csv_path]
            if "URLEXCEPTION" in entry:
                raise entry["URLEXCEPTION"]
            if "URLDATA" in entry:
                fd = io.StringIO(entry["URLDATA"])
            else:
                if do_not_cache:
                    raise RuntimeError("Need to fetch URL manually.")
                try:
                    url = generate_presigned_url(self.csv_path)
                    url_fd = urllib.request.urlopen(url)
                except Exception as e:
                    entry["URLEXCEPTION"] = e
                    raise e
                fd = io.StringIO()
                while True:
                    text = url_fd.read()
                    if isinstance(text, bytes):
                        text = text.decode()
                    if len(text) == 0:
                        break
                    fd.write(text)
                fd.seek(0)
                entry["URLDATA"] = fd.getvalue()
            return fd
        else:
            return open(self.csv_path, "rt")

    def get_cache_info(self):
        """Get the cached information for the data file"""
        global header_cache
        entry = header_cache.get(self.csv_path, dict(ctime=0))
        if is_url_path(self.csv_path):
            if self.csv_path not in header_cache:
                header_cache[self.csv_path] = entry
            return entry
        ctime = os.stat(self.csv_path).st_ctime
        if ctime > entry["ctime"]:
            entry = header_cache[self.csv_path] = {}
            entry["ctime"] = ctime
        return entry

    def get_header(self, do_not_cache=False):
        """Read the header fields from the csv file

        Open the csv file indicated by the settings and read the fields
        of its first line. These should be the measurement columns.
        """
        entry = self.get_cache_info()
        if "header" in entry:
            return entry["header"]

        fd = self.open_csv(do_not_cache=do_not_cache)
        reader = csv.reader(fd)
        header = next(reader)
        fd.close()
        entry["header"] = [header_to_column(column) for column in header]
        return entry["header"]

    def validate_module(self, pipeline):
        csv_path = self.csv_path

        try:
            self.open_csv()
        except IOError as e:
            import errno

            if e.errno == errno.EWOULDBLOCK:
                raise ValidationError(
                    "Another program (Excel?) is locking the CSV file %s."
                    % self.csv_path,
                    self.csv_file_name,
                )
            else:
                raise ValidationError(
                    "Could not open CSV file %s (error: %s)" % (self.csv_path, e),
                    self.csv_file_name,
                )

        try:
            self.get_header()
        except Exception as e:
            raise ValidationError(
                "The CSV file, %s, is not in the proper format."
                " See this module's help for details on CSV format. (error: %s)"
                % (self.csv_path, e),
                self.csv_file_name,
            )    

    def run(self, workspace):
        print("==== RUNNING ====")
        print('Output_directory:')
        print(str(get_default_image_directory()))

        top_level_window = workspace.frame

        # Load dataframes and numpy arrays from Module1
        df_path = workspace.measurements.get_current_image_measurement("OutlierDetection_df_path")
        df = pd.read_csv(df_path)

        df_2_clean_path = workspace.measurements.get_current_image_measurement("OutlierDetection_df_2_clean_path")
        df_2_clean = pd.read_csv(df_2_clean_path)

        labels_2_path = workspace.measurements.get_current_image_measurement("OutlierDetection_labels_2_path")
        labels_2 = pd.read_csv(labels_2_path)

        X_pca_df_path = workspace.measurements.get_current_image_measurement("OutlierDetection_X_pca_df_path")
        X_pca_df = pd.read_csv(X_pca_df_path)

        X_scaled_path = workspace.measurements.get_current_image_measurement("OutlierDetection_X_scaled_path")
        X_scaled = np.load(X_scaled_path,allow_pickle=True)

        scaler = StandardScaler()

        # loading data frames/numpy arrays from Module1

        print('==== LOADED Module1 DATA ====')
        #Module1_csv_path = workspace.pipeline._Pipeline__Modules
        
        if self.pick_outliers:
            final_string = str(self.outliers_list)
        else:
            initial_string = workspace.measurements.get_current_image_measurement("OutlierDetection_ResultValue")
            dlg = wx.TextEntryDialog(
                #None,
                top_level_window,
                "Review and modify the outliers",
                "Edit Outlier Data",
                value=initial_string
            )

            if dlg.ShowModal() == wx.ID_OK:
                final_string = dlg.GetValue()
                dlg.Destroy()
            else:
                final_string = initial_string

        outliers_int_array = []

        for outlier in final_string.split(','):
            outliers_int_array.append(int(outlier))

        # removing rows corresponding to outliers from data frame
        # user inputting row numbers corresponding to outliers to remove from data frame
        print(X_pca_df.iloc[outliers_int_array]) # values input by user

        # turning both labels and df into numpy arrays
        df_2_clean_np = df_2_clean.to_numpy()
        labels_3 = labels_2.to_numpy(dtype = str)

        df_2_clean_cleaned = np.delete(df_2_clean_np,[], axis=0)
        labels_cleaned = np.delete(labels_3,[], axis=0)

        # redo standardization
        scaler = StandardScaler()
        X_scaled_cleaned = scaler.fit_transform(df_2_clean_cleaned)

        # redo PCA
        pca_2 = PCA()
        X_pca_cleaned = pca_2.fit_transform(X_scaled_cleaned)

        eigenvalues_2 = pca_2.explained_variance_
        print(eigenvalues_2)

        # applying Kaiser criterion
        n_components2 = np.sum(eigenvalues_2 > 1)
        print("Number of components to keep:", n_components2)

        # keeping only PCs according to Kaiser criterion
        # extracting first 13 components
        X_pca_cleaned2 = X_pca_cleaned[:,0:n_components2]
        print(X_pca_cleaned2.shape)

        # convert numpy array to pandas data frame
        X_pd = pd.DataFrame(X_pca_cleaned2)

        # hierarchical clustering
        linked = linkage(X_pd, method = "ward", metric = "euclidean")

        # create a dendrogram to visualize figures
        den_figure = plt.figure(figsize = (10,5))

        dn1 = dendrogram(linked,
                orientation = "top",
                labels = X_pd.index,
                distance_sort = "descending",
                show_leaf_counts = True,
                color_threshold=0,
                above_threshold_color='black')
        
        print()
        # plt title
        plt.xlabel("instances")
        plt.ylabel("Ward distance")
        plt.show()
        # dendrogram displaying data structure generated as output of module 2
        # user decides based on figure how many clusters to choose as input for module 3

        den_num_clusters = '0'

        den_num_clusters_dlg = wx.TextEntryDialog(
            None,
            "Cluster Threshold",
            "Enter the Cluster Threshold number:",
            value=den_num_clusters
        )

        if den_num_clusters_dlg.ShowModal() == wx.ID_OK:
            num_clusters = den_num_clusters_dlg.GetValue()
            workspace.measurements.add_image_measurement("OutlierAndClusterSettings" + "_num_clusters", num_clusters)

        den_num_clusters_dlg.Close()
        plt.close()
        
        def output_df_to_csv(modulename,df,name):
            path = get_default_output_directory() + '/' + modulename + '_' + name + '.csv'
            df.to_csv(path, index=False)
            workspace.measurements.add_image_measurement(modulename + "_" + name + "_path", path)

        def output_np_to_csv(modulename,nparray,name):
            path = get_default_output_directory() + '/' + modulename + '_' + name + '.npy'
            np.save(path, nparray)
            workspace.measurements.add_image_measurement(modulename + "_" + name + "_path", path)

        output_df_to_csv(self.module_name,df,'df')
        output_np_to_csv(self.module_name,labels_cleaned,'labels_cleaned')
        output_df_to_csv(self.module_name,X_pca_df, 'X_pca_df')
        output_df_to_csv(self.module_name,X_pd,'X_pd')
        output_np_to_csv(self.module_name,X_scaled, 'X_scaled')
        output_np_to_csv(self.module_name,linked, 'linked')        

        print('=====END OF OutlierAndClusterSettings=====')