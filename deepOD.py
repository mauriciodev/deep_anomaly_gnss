from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd
import torch
import nni
import sklearn

gnss_data_path = 'data/BRAZ_train.csv.csv'
#data = np.genfromtxt(gnss_data_path, delimiter=',', skip_header=1, dtype=None)
df = pd.read_csv(gnss_data_path)
data = df.to_numpy()
print(data.shape)
X_train = data[:, [1,2]]
print(X_train.shape)
X_test = X_train

truth = data[:, [4]]





params = {'seq_len': 100,
          'epochs':20,
          'd_model': 128
          
    }
params.update( nni.get_next_parameter() )
print(params)
# time series anomaly detection methods
from deepod.models.time_series import TimesNet as Model
model = Model(**params) 

#from deepod.models.time_series import AnomalyTransformer as Model
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
"""model = Model(seq_len=10, stride=1, epochs=20,
              batch_size=32, k=20, lr=1e-4,
              device=device, random_state=42)"""



#parameters = model.fit_auto_hyper(X_train)
#scores = model.decision_function(X_train)
model.fit(X_train)
pred = model.predict(X_train)
metrics = sklearn.metrics.precision_recall_fscore_support(pred.flatten(), truth.flatten())
accuracy = sklearn.metrics.accuracy_score(pred.flatten(), truth.flatten())
f1 = sklearn.metrics.f1_score(pred.flatten(), truth.flatten())
nni.report_final_result(f1)
print(f"Accuracy {accuracy}")
print(f"F1 score {f1}")

#df.plot(x='gps_week', y=['dn(m)', 'de(m)'])
#plt.bar(df['gps_week'], scores/max(scores)*X_train.max(), label = 'scores')
#plt.plot(df['gps_week'], pred[0], label = 'guess', linestyle='', marker='o')


#pred_df = df['dn(m)'].to_numpy().copy()
#pred_df[pred==0] = np.nan
#plt.vlines(df['gps_week'][pred==1].to_numpy(), ymin=X_train.min(), ymax=X_train.max())

#plt.legend()
