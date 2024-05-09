import pandas as pd
from matplotlib import pyplot as plt
from darts import TimeSeries
from darts.ad import KMeansScorer
from sklearn.preprocessing import MinMaxScaler

station = 'BRAZ'

gnss_data_path = f'dataset/{station}/{station}_NEU_train.csv'
gnss_label_path = f'dataset/{station}/{station}_NEU_test_label.csv'

gnss_data = pd.read_csv(gnss_data_path)
gnss_label = pd.read_csv(gnss_label_path)

train = TimeSeries.from_dataframe(df=gnss_data, time_col=gnss_data.columns[0], value_cols=gnss_data.columns[1:3], fill_missing_dates=True, freq=1, fillna_value=0.0)
label = TimeSeries.from_dataframe(df=gnss_label, time_col=gnss_label.columns[0], value_cols=gnss_label.columns[1], fill_missing_dates=True, freq=1, fillna_value=0.0)

scorer = KMeansScorer(k=2, window=1)
scorer.fit(train)
scores = scorer.score(train)
scores = scores.data_array().to_numpy().squeeze()

# Scaling scores to 0-1
scaler = MinMaxScaler(feature_range=(0, 1))
scores = scaler.fit_transform(scores.reshape(-1, 1)).flatten()

# Create the figure and primary y-axis
fig, ax1 = plt.subplots()

# Plot GNSS data on the primary y-axis
ax1.plot(train.time_index, train.values()[:, 0], color = 'cornflowerblue', label = 'Series DN')
ax1.plot(train.time_index, train.values()[:, 1], color = 'gold', label = 'Series DE')
#ax1.plot(train.time_index, train.columns[0], color = 'magenta', label = 'Series DU')

# Plotting anomalies
#anomalies = gnss_label[gnss_label.label == 1]
#ax1.vlines(anomalies.gps_week, ymin=plt.ylim()[0], ymax=plt.ylim()[1], color = 'black', linestyle='dashed', alpha=0.5, label='Descontinuity')

# Plotting predictions
#predictions = gnss_label[gnss_label.pred == 1]
#ax1.vlines(predictions.gps_week, ymin=plt.ylim()[0], ymax=plt.ylim()[1], color = 'red', alpha=0.5, label='Prediction')

# Create the secondary y-axis (twinx)
ax2 = ax1.twinx()

# Plot data3 on the secondary y-axis
ax2.plot(train.time_index, scores, color='black', linewidth=0.5, label='Scores')

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
plt.title(f'Station: {station}', loc='center')
plt.show()