import pandas as pd

def get_quakes():
    return pd.read_csv(f'dataset/earthquakes_ecuador.csv')

def get_gps_week(time_series:pd.Series) -> pd.Series:
    gps_t0 = pd.to_datetime('1980-01-06T00:00:00.000Z')
    return ((time_series-gps_t0).dt.days/7).astype(int)

def process_quakes(quakes_df):
    quakes_df.time = pd.to_datetime(quakes_df.time, errors='coerce')
    quakes_df['gps_week'] = get_gps_week(quakes_df.time)
    quakes_df.to_csv(f'dataset/quakes.csv', index=False, header=True)

if __name__ == '__main__':
    process_quakes(pd.read_csv(f'dataset/earthquakes_ecuador.csv'))
