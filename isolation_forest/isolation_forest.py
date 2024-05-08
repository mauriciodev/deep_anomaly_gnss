from sklearn.ensemble import IsolationForest as Model
import pandas as pd
from matplotlib import pyplot as plt

station = 'BRAZ'

gnss_data_path = f'dataset/{station}/{station}_NEU_train.csv'
gnss_label_path = f'dataset/{station}/{station}_NEU_test_label.csv'

gnss_data = pd.read_csv(gnss_data_path)
gnss_label = pd.read_csv(gnss_label_path)

model = Model(n_estimators=200, random_state = 42, contamination = 0.1)

training_data = gnss_data.iloc[:, [1,2]]
model.fit(training_data)

scores = model.decision_function(training_data)
pred = model.predict(training_data)

plt.plot(gnss_data.gps_week, gnss_data['dn(m)'], color = 'cornflowerblue', label = 'Series DN')
plt.plot(gnss_data.gps_week, gnss_data['de(m)'], color = 'gold', label = 'Series DE')
plt.plot(gnss_data.gps_week, gnss_data['du(m)'], color = 'magenta', label = 'Series DU')

plt.plot(gnss_data.gps_week, scores, color='black', linewidth=0.5, label='Scores')

#gnss_label['pred'] = pred
#predictions = gnss_label[gnss_label.pred == 1]
#plt.vlines(predictions.gps_week, ymin=plt.ylim()[0], ymax=plt.ylim()[1], color = 'red', alpha=0.5, label='Prediction')

plt.legend()
plt.show()