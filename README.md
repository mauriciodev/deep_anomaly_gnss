## deep_anomaly_gnss
deep_anomaly_gnss provides a Python code that allows:
1. Download and preprocess of SIRGAS weekly solutions for anomaly detection tasks
2. Anomaly detection in SIRGAS preprocessed weekly solutions with [TimesNet](https://arxiv.org/abs/2210.02186) using early stop
3. Anomaly detection in SIRGAS preprocessed weekly solutions with [Darts](https://unit8co.github.io/darts/index.html).
4. Darts used models:
   - Gaussian Process
   - Kalman
   - Moving Average
   - TSMixer
   - Transformer
5. Darts used anomaly scorers:
   - Norm
   - KMeans
   - Difference 

## Requirements
1. PyTorch
2. [Darts](https://unit8co.github.io/darts/index.html)
3. [DeepOD](https://github.com/xuhongzuo/DeepOD/tree/main)
4. [NNI](https://nni.readthedocs.io)
5. Matplotlib, Scikit-learn, Numpy, Pandas  

pip install u8darts[torch] nni==2.10.1  
pip install deepod==0.4.1 --no-deps  

## Stations selection
To download SIRGAS weekly solutions you need to insert the four digit code of the desired stations in a txt file (e.g. ALAR, AMCO, BABR).  
Save the file in the dataset folder.

Our research used 95 stations (77 from Brazil and 18 from Ecuador) whose files are in the dataset folder.

## Downloading data from both stations (Brazil and Ecuador)
python preprocess/gnss_preprocessor.py -s dataset/brazil_stations.txt  
python preprocess/gnss_preprocessor.py -s dataset/ecuador_stations.txt  

## Hyperparameter tunning for BRAZ and CHEC stations
python nni_experiment/nni_config.py -s BRAZ  
python nni_experiment/nni_config.py -s CHEC  

## Processing every station in each country with TimesNet and the specified params
python process_stations/process_stations.py -s dataset/brazil_stations.txt -p process_stations/params_chec.json  
python process_stations/process_stations.py -s dataset/ecuador_stations.txt -p process_stations/params_chec.json  

python process_stations/process_stations.py -s dataset/brazil_stations.txt -p process_stations/params_braz.json  
python process_stations/process_stations.py -s dataset/ecuador_stations.txt -p process_stations/params_braz.json  

## Processing every station in each country with Darts defining model index (0, 1, 2, 3, 4) and scorer index (0, 1, 2)
python process_stations/process_stations.py -s dataset/brazil_stations.txt -mi 2 -si 1  
python process_stations/process_stations.py -s dataset/ecuador_stations.txt -mi 2 -mi 1  
