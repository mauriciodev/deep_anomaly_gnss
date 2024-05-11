import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler

# TSlearn imports
from tslearn.svm import TimeSeriesSVC
from tslearn.clustering import TimeSeriesKMeans

class TSlearnTrainer():
    def __init__(self, model, station:str, use_du:bool) -> None:
        self.station = station
        self.use_du = use_du

        # Defining station filepaths
        self.gnss_data_path = f'dataset/{station}/{station}_NEU_train.csv'
        self.gnss_label_path = f'dataset/{station}/{station}_NEU_test_label.csv'
        self.png_path = f'dataset/{station}/{station}_trained_tslearn.png'
        self.metrics_path = f'dataset/{station}/{station}_metrics_tslearn.json'

        # Getting the data
        self.gnss_data, self.gnss_label = self.get_data(self.gnss_data_path, self.gnss_label_path)

        # Setting our model
        self.model = model

    def train(self):
        # Defining training data
        if self.use_du:
            training_data = self.gnss_data
        else:
            training_data = self.gnss_data.iloc[:, [1,2]]

        # Defining label data
        label_data = self.gnss_label.iloc[:, 0]

        # Training
        start = time.time()
        self.model.fit(training_data)
        end = time.time()

        # Elapsed fit time
        fit_time = end - start

        # Decision score
        start = time.time()
        scores = self.model.predict(training_data)
        end = time.time()

        # Elapsed score time
        score_time = end - start

        scaler = MinMaxScaler(feature_range=(0, 1))
        scores = scaler.fit_transform(scores.reshape(-1, 1)).flatten()

        scores = scores[:len(self.gnss_data.gps_week)]

        print(f"Elapsed time to fit: {fit_time:.2f} seconds")
        print(f"Elapsed time to score: {score_time:.2f} seconds")

        return scores
    
    def get_data(self, gnss_data_path: str, gnss_label_path:str) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            gnss_data = pd.read_csv(gnss_data_path)
            gnss_label = pd.read_csv(gnss_label_path)
        except:
            gnss_data, gnss_label = pd.DataFrame(), pd.DataFrame()

        return gnss_data, gnss_label    

    def plot_experiment(self, scores:np.array) -> None:
        plt.clf()

        # Create the figure and primary y-axis
        fig, ax1 = plt.subplots()

        # Plot GNSS data on the primary y-axis
        ax1.plot(self.gnss_data.gps_week, self.gnss_data['dn(m)'], color = 'cornflowerblue', label = 'Series DN')
        ax1.plot(self.gnss_data.gps_week, self.gnss_data['de(m)'], color = 'gold', label = 'Series DE')
        ax1.plot(self.gnss_data.gps_week, self.gnss_data['du(m)'], color = 'magenta', label = 'Series DU')

        # Plotting anomalies
        anomalies = self.gnss_label[self.gnss_label.label == 1]
        if not anomalies.empty:
            ax1.vlines(anomalies.gps_week, ymin=plt.ylim()[0], ymax=plt.ylim()[1], color = 'black', linestyle='dashed', alpha=0.5, label='Descontinuity')

        # Create the secondary y-axis (twinx)
        ax2 = ax1.twinx()

        # Plot data3 on the secondary y-axis
        ax2.plot(self.gnss_data.gps_week, scores, color='black', linewidth=0.5, label='Scores')

        # Set labels for axes
        ax1.set_xlabel('GPS Week')
        ax1.set_ylabel('Deviation in meters (m)')
        ax2.set_ylabel('Normalized Score [0-1]')

        # Add legend using all lines
        # ask matplotlib for the plotted objects and their labels
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)

        # Save the plot
        plt.title(f'Station: {self.station}', loc='center')
        plt.savefig(self.png_path, format='png')

if __name__ == '__main__':
    station = 'BRAZ'
    #model = TimeSeriesSVC()
    model = TimeSeriesKMeans(n_clusters=20)
    trainer = TSlearnTrainer(model=model, station=station, use_du=False)
    scores = trainer.train()

    trainer.plot_experiment(scores)