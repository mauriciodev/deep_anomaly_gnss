import os
from tqdm import tqdm
import urllib.request
import pandas as pd
import re
from io import StringIO
from matplotlib import pyplot as plt
import numpy as np

class GNSSPreprocessor():
    def __init__(self, stations:list=['BRAZ'], destination:str='/kaggle/working', update:bool=False):
        self.stations = stations
        self.destination = destination
        self.update = update
        
    def download(self, station_folder:str, url: str) -> str:
        filename = os.path.basename(url)
        filepath=os.path.join(station_folder, filename)
        
        if os.path.exists(filepath) and not self.update:
            print(f'File already exists: {filepath}')
        else:
            print(f"Downloading {url} -> {filepath}")
            try:
                with open(filepath, "wb") as f:
                    with tqdm(unit="B", unit_scale=True, desc=filename) as pbar:
                        response = urllib.request.urlopen(url)
                        chunk_size = 1024
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            pbar.update(len(chunk))
            except urllib.error.URLError as e:
                print(f"Download failed: {url} - {e}")
        return filepath
    
    def download_sirgas(self) -> None:
        for station in self.stations:
            station_folder = os.path.join(self.destination, station)
            os.makedirs(station_folder, exist_ok=True)
            
            xyz_url = f'https://www.sirgas.org/fileadmin/docs/SIRGAS_CRD/{station}.XYZ'
            neu_url = f'https://www.sirgas.org/fileadmin/docs/SIRGAS_TS/{station}.NEU'
            dsc_url = f'https://www.sirgas.org/fileadmin/docs/SIRGAS_TS/{station}.dsc'
            
            # Downloading data
            xyz_filepath = self.download(station_folder, xyz_url)
            neu_filepath = self.download(station_folder, neu_url)
            dsc_filepath = self.download(station_folder, dsc_url)
            
            # Reading NEU file into dataframes
            df, dsc_df = self.read_NEU(neu_filepath)

            if df is None:
                print(f"Error processing station {station}.")
                continue #skip only this station
            
            # Preprocessing dataframes
            neu_df_train, neu_df_test_label = self.preprocess_NEU(df, dsc_df)
            
            # Saving the dataframes
            neu_df_train_filename = f'{station}_NEU_train.csv'
            neu_df_train_filepath = os.path.join(station_folder,neu_df_train_filename)
            self.save_dataframe(neu_df_train_filepath, neu_df_train)

            neu_df_test_label_filename = f'{station}_NEU_test_label.csv'
            neu_df_test_label_filepath = os.path.join(station_folder,neu_df_test_label_filename)
            self.save_dataframe(neu_df_test_label_filepath, neu_df_test_label)
            
            # Saving the plot
            plot_filename = f'{station}.png'
            plot_filepath = os.path.join(station_folder, plot_filename)
            self.save_plot(station, neu_df_train, neu_df_test_label, plot_filepath)
            
    def save_plot(self, station:str, neu_df_train:pd.DataFrame, neu_df_test_label:pd.DataFrame, plot_filepath:str) -> None:
        plt.clf()
        #plotting data
        plt.plot(neu_df_train.gps_week, neu_df_train['dn(m)'], label = 'Series DN')
        plt.plot(neu_df_train.gps_week, neu_df_train['de(m)'], label = 'Series DE')
        plt.plot(neu_df_train.gps_week, neu_df_train['du(m)'], label = 'Series DU')
        
        # Plotting anomalies
        anomalies = neu_df_test_label[neu_df_test_label.label == 1]
        plt.vlines(anomalies.gps_week, ymin=plt.ylim()[0], ymax=plt.ylim()[1], color = 'black', alpha=0.5, label='Descontinuity')

        plt.legend()
        plt.title(f'Station: {station}', loc='center')
        plt.savefig(plot_filepath, format='png')
            
    def read_NEU(self, filepath:str) -> tuple[pd.DataFrame, pd.DataFrame]:
        with open(filepath,'r', encoding='ISO-8859-15') as a:
            s = a.read()
            # Reading only the data that is between the  -------- lines 
            try:
                data = re.search(r'------\n(.*)\n-----', s, flags=re.DOTALL).group(1)
            except AttributeError:
                data = None
            
            # Reading the discontinuities
            try:
                dsc = re.search(r'Discontinuities detected at this station:\n(.*)\n\n', s, flags=re.DOTALL).group(1)
            except AttributeError:
                dsc = None

        inputHeaders = ["yyyy.yyyy",'civil_date', 'gps_week', 'geoframe', "station_id", "station_dome", "dn(m)", "de(m)","du(m)", "sig_dn(m)", "sig_de(m)", "sig_du(m)"] 

        if dsc is not None and 'None' not in dsc:
            dsc_df = pd.read_fwf(StringIO(dsc), widths=[13,67], header=None, parse_dates=True, names=['civil_date', 'motive'])
            dsc_df.civil_date = pd.to_datetime(dsc_df.civil_date, format='%Y-%m-%d', errors='coerce')
            dsc_df.dropna(inplace=True)
        else:
            dsc_df = None

        if data is not None and 'None' not in data:
            df = pd.read_csv(StringIO(data),  sep='\s+', names=inputHeaders)
            df.civil_date = pd.to_datetime(df.civil_date, format='%Y-%m-%d', errors='coerce')
            df.dropna(inplace=True)
        else:
            df = None

        return df, dsc_df
    
    def save_dataframe(self, filepath:str, df:pd.DataFrame) -> None:
        df.to_csv(filepath, index=False, header=True)
        
    def get_gps_week(self, time_series:pd.Series) -> pd.Series:
        gps_t0 = pd.to_datetime('1980-01-06T00:00:00')
        return (np.ceil((time_series-gps_t0).dt.days/7)).astype(int)
    
    def preprocess_NEU(self, neu_df:pd.DataFrame, dsc_df:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Adding a label column
        neu_df['label'] = 0
        
        if dsc_df is not None:
            # Calculating the values for GPS Week for the civil dates of our discontinuities
            gps_week = self.get_gps_week(dsc_df.civil_date)
        
            # Setting the labels to 1 for those gps weeks
            neu_df.loc[neu_df['gps_week'].isin(gps_week), 'label'] = 1
        
        # Columns to be removed
        column_names = ['yyyy.yyyy', 'civil_date', 'geoframe', 'station_id', 'station_dome', 'sig_dn(m)', 'sig_de(m)', 'sig_du(m)']
        label_column = ['label']
        data_columns = ['dn(m)', 'de(m)', 'du(m)']
        
        # Keeping just what is needed for training with Anomaly Transformer
        neu_df_train = neu_df.drop(column_names+label_column, axis=1)
        
        # Keeping just what is need for testing
        neu_df_test_label = neu_df.drop(column_names+data_columns, axis=1)
        
        return neu_df_train, neu_df_test_label

def read_stations_file(stations_file:str) -> list:
    try:
        with open(stations_file) as file:
            content = file.read()
            content = content.replace(' ', '').strip()
            return content.split(',')
    except FileNotFoundError as e:
        print(f"Error: File '{stations_file}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file '{stations_file}': {e}")
        return []

def exec_preprocess():
    # Brazilian Stations
    br_stations_filepath = 'dataset/brazil_stations.txt'
    ec_stations_filepath = 'dataset/ecuador_stations.txt'
    br_stations = read_stations_file(br_stations_filepath)
    ec_stations = read_stations_file(ec_stations_filepath)

    stations = br_stations + ec_stations

    # Execute download and preprocess
    destination = 'dataset'
    '''
    # Downloading data
        1. Download
        2. Pre-process
        3. Save pre-processed CSV
        4. Save plot
    '''
    preprocessor = GNSSPreprocessor(stations=stations, destination=destination)
    preprocessor.download_sirgas()
    
if __name__ == '__main__':
    exec_preprocess()