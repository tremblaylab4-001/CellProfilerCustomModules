import os
import io
import urllib
import csv

# CellProfiler Imports
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
#from cellprofiler_core.setting import Binary
from cellprofiler_core.setting import ValidationError
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
import wx as wx

"""Cache of header columns for files"""
header_cache = {}

class OutlierDetection(cpm.Module):
    # 1. Basic Metadata
    module_name = "OutlierDetection"
    variable_revision_number = 1
    category = "Utility"

    def create_settings(self):
        # 2. Define user-configurable settings
        #self.welcome_message = cps.text.Text(
        #    "Welcome Message", "Hello from CellProfiler!", 
        #    doc="This message will print to the console."
        #)

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
        return [self.csv_directory, self.csv_file_name]

    def visible_settings(self):
        # Return settings visible in the GUI
        return [self.csv_directory, self.csv_file_name]

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

        if self.csv_directory.dir_choice != URL_FOLDER_NAME:
            if not os.path.isfile(csv_path):
                raise ValidationError(
                    "No such CSV file: %s" % csv_path, self.csv_file_name
                )

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

        # loading data frame
        df = pd.read_csv(self.csv_path)

        #self.module_num-1 
        #prev_module = workspace.pipeline["(self.module_num-1)"]
        
        # transforming data
        # selecting parameters to include
        labels_2 = df.loc[:,["ObjectNumber", "FileName_Blind", "Condition"]]

        df_2 = df.loc[:,["AreaShape_Area",
                        "AreaShape_BoundingBoxArea", 
                        "AreaShape_Center_X",
                        "AreaShape_Center_Y",
                        "AreaShape_CentralMoment_0_0",
                        "AreaShape_CentralMoment_0_1",
                        "AreaShape_CentralMoment_0_2",
                        "AreaShape_CentralMoment_0_3",
                        "AreaShape_CentralMoment_1_0",
                        "AreaShape_CentralMoment_1_1",
                        "AreaShape_CentralMoment_1_2",
                        "AreaShape_CentralMoment_1_3",
                        "AreaShape_CentralMoment_2_0",
                        "AreaShape_CentralMoment_2_1",
                        "AreaShape_CentralMoment_2_2",
                        "AreaShape_CentralMoment_2_3",
                        "AreaShape_Compactness",
                        "AreaShape_ConvexArea",
                        "AreaShape_Eccentricity",
                        "AreaShape_EquivalentDiameter",
                        "AreaShape_Extent",
                        "AreaShape_HuMoment_0",
                        "AreaShape_HuMoment_1",
                        "AreaShape_HuMoment_2",
                        "AreaShape_HuMoment_3",
                        "AreaShape_HuMoment_4",                  
                        "AreaShape_HuMoment_5",
                        "AreaShape_HuMoment_6",
                        "AreaShape_InertiaTensorEigenvalues_0",
                        "AreaShape_InertiaTensorEigenvalues_1",
                        "AreaShape_InertiaTensor_0_0",
                        "AreaShape_InertiaTensor_0_1",
                        "AreaShape_InertiaTensor_1_0",
                        "AreaShape_InertiaTensor_1_1",
                        "AreaShape_MajorAxisLength",
                        "AreaShape_MaxFeretDiameter",
                        "AreaShape_MeanRadius",
                        "AreaShape_MinFeretDiameter",
                        "AreaShape_MinorAxisLength",
                        "AreaShape_NormalizedMoment_0_2",
                        "AreaShape_NormalizedMoment_0_3",
                        "AreaShape_NormalizedMoment_1_1",
                        "AreaShape_NormalizedMoment_1_2",
                        "AreaShape_NormalizedMoment_1_3",
                        "AreaShape_NormalizedMoment_2_0",
                        "AreaShape_NormalizedMoment_2_1",
                        "AreaShape_NormalizedMoment_2_2",
                        "AreaShape_NormalizedMoment_2_3",
                        "AreaShape_NormalizedMoment_3_0",
                        "AreaShape_NormalizedMoment_3_1",
                        "AreaShape_NormalizedMoment_3_2",
                        "AreaShape_NormalizedMoment_3_3",
                        "AreaShape_Orientation",
                        "AreaShape_Perimeter",
                        "AreaShape_Solidity",
                        "AreaShape_SpatialMoment_0_0",
                        "AreaShape_SpatialMoment_0_1",
                        "AreaShape_SpatialMoment_0_2",
                        "AreaShape_SpatialMoment_0_3",
                        "AreaShape_SpatialMoment_1_0",
                        "AreaShape_SpatialMoment_1_1",            
                        "AreaShape_SpatialMoment_1_2",
                        "AreaShape_SpatialMoment_1_3",
                        "AreaShape_SpatialMoment_2_0",
                        "AreaShape_SpatialMoment_2_1",
                        "AreaShape_SpatialMoment_2_2",
                        "AreaShape_SpatialMoment_2_3",
                        "ObjectSkeleton_NumberBranchEnds_MorphBlue",
                        "ObjectSkeleton_NumberNonTrunkBranches_MorphBlue",
                        "ObjectSkeleton_NumberTrunks_MorphBlue",
                        "ObjectSkeleton_TotalObjectSkeletonLength_MorphBlue"]]

        # excluding infinite and na values
        df_2_clean = df_2.loc[:, ~(np.isinf(df_2) | df_2.isna()).any()]

        # scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_2_clean)

        # run PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)

        # getting eigenvalues
        eigenvalues = pca.explained_variance_
        print(eigenvalues)

        # keep only eigenvalues larger than 1
        n_components = np.sum(eigenvalues > 1)
        print("Number of components to keep:", n_components)
        # output should be used as input into next module

        # selecting columns 0 to 14, including 14
        X_pca_components = X_pca[:,0:n_components]
        #print(X_pca_compnents.head())

        #setting seed
        np.random.seed(42)

        # convert to numpy array
        matrix = np.array(X_pca_components)

        # compute LOF
        lof = LocalOutlierFactor(n_neighbors=90)
        labels = lof.fit_predict(matrix)

        # Each point is compared to its k nearest neighbors.
        # LOF calculates a local density for the point and for its neighbors.
        # LOF scores (negative by sklearn convention)
        # labels = lof.fit_predict(X)  # 1 = inlier, -1 = outlier
        lof_scores = -lof.negative_outlier_factor_

        # define threshold
        # LOF ≈ 1 → the point has similar density as neighbors → normal
        # LOF > 1 → the point has lower density than neighbors → potential outlier
        threshold = 3
        
        # select outliers
        outliers = X_pca_components[lof_scores > threshold]

        X_pca_df = pd.DataFrame(X_pca_components)

        X_pca_df["Outlier"] = np.where(
            lof_scores > threshold,
            "Outlier",
            "Inlier"
        )

        # filtering data frame by column names
        outliers = X_pca_df.Outlier == "Outlier"
        all_outliers = X_pca_df.Outlier

        # generating figure displaying outliers
        print("Number of components to keep:", n_components)
        X_pca_df["PC1"] = X_pca_df.iloc[:,0]
        X_pca_df["PC2"] = X_pca_df.iloc[:,1]
        X_pca_df["PC3"] = X_pca_df.iloc[:,2]
        print(X_pca_df.columns)

        fig = go.Figure()
        color_map = {
            "Outlier": "red",
            "Inlier": "black"
        }

        colors = X_pca_df["Outlier"].map(color_map)
        scatter = go.Scatter3d( x = X_pca_df["PC1"], 
                                y = X_pca_df["PC2"], 
                                z = X_pca_df["PC3"], 
                                mode = "markers", 
                                marker = dict(size = 8, color = colors),
                                customdata=labels_2[["FileName_Blind", "ObjectNumber"]].values,
                                hovertemplate=("Image = %{customdata[0]}<br>" +
                                               "Object = %{customdata[1]}<br>" ))
        
        fig.add_trace(scatter)
        fig.layout.title.text = "Outliers"
        dir(go)
        fig.show()

        # adding outliers column to data frame
        X_pca_df.index = range(1, len(X_pca_df) + 1)
        outliers = X_pca_df[X_pca_df["Outlier"] == "Outlier"]
        print(outliers) # row numbers start at 0 now, +1 should be added so that it corresponds to input data frame
        df_with_outliers = pd.concat([labels_2,all_outliers], axis = 1)
        df_with_outliers.to_csv("df_with_outliers_Sep1_2.csv")   # user receives this csv as output at this point
        # output generated by module 1
        # row index of all outlying values
        # user should select which values to keep
        # values chosen by user should be taken as input for module 2
        # end of module 1

        outliers_string = ""

        for i in range(0, len(outliers.index.values), 1):
            outliers_string += str(outliers.index.values[i])
            if i < (len(outliers.index.values)-1):
                outliers_string += ","
            
        # end of module 1

        workspace.measurements.add_image_measurement("OutlierDetection_ResultValue", outliers_string)

        def output_df_to_csv(modulename,df,name):
            self.path = get_default_output_directory() + '/' + modulename + '_' + name + '.csv'
            print('Writing ' + self.path + '.')
            df.to_csv(self.path, index=False)
            workspace.measurements.add_image_measurement(modulename + "_" + name + "_path", self.path)

        def output_np_to_csv(modulename,nparray,name):
            self.path = get_default_output_directory() + '/' + modulename + '_' + name + '.npy'
            print('Writing ' + self.path + '.')
            np.save(self.path, nparray)
            workspace.measurements.add_image_measurement(modulename + "_" + name + "_path", self.path)

        output_df_to_csv(self.module_name,df,'df')
        output_df_to_csv(self.module_name,df_2_clean,'df_2_clean')
        output_df_to_csv(self.module_name,labels_2,'labels_2')
        output_df_to_csv(self.module_name,X_pca_df,'X_pca_df')
        output_np_to_csv(self.module_name,X_scaled,'X_scaled')

        # Variables needed in next module
        # df
        # scaler = StandardScaler()
        # X_scaled
        # n_components
        # 
