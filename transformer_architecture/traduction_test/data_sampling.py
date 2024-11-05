import pandas as pd
import gcsfs
import random
import argparse

from transformer_architecture.configs.confs import load_conf, clean_params

main_params = load_conf(include=True)
main_params = clean_params(main_params)

bucket_path = main_params['bucket_path']
DATASET_PROPORTION = main_params['DATASET_PROPORTION']
EXPERIENCE_NAME = main_params['EXPERIENCE_NAME']

local_output_path = f'data/sampled_data_{EXPERIENCE_NAME}.csv'  

dataframe_size=22520376

sample_size = int(dataframe_size * DATASET_PROPORTION)

skip_rows = sorted(random.sample(range(1, dataframe_size + 1), dataframe_size - sample_size))

df_sample = pd.read_csv(bucket_path, skiprows=skip_rows)

df_sample.to_csv(local_output_path, index=False)
print(f"Échantillon enregistré en local sous : {local_output_path}")
