import pandas as pd


def load_data(name, column_names=('julian_day', 'noiseless_flux', 'noisy_flux', 'sigma', 'deviation_used'),
              appendix='.flux'):
    dataframe = pd.read_csv('data/' + name + appendix, delimiter='\s+', names=column_names)
    return dataframe
