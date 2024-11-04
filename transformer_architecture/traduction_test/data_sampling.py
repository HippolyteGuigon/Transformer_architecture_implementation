import pandas as pd
import gcsfs
import random
from datetime import datetime

print("STARTED")
current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

bucket_path = 'gs://french-english-raw-data/en-fr.csv'  
fraction = 0.01  
local_output_path = f'transformer_architecture/data/sampled_data_{current_datetime}.csv'  

dataframe_size=22000000

print("STARTED2")

sample_size = int(dataframe_size * fraction)

skip_rows = sorted(random.sample(range(1, dataframe_size + 1), dataframe_size - sample_size))

print("STARTED3")

df_sample = pd.read_csv(bucket_path, skiprows=skip_rows)

df_sample.to_csv(local_output_path, index=False)
print(f"Échantillon enregistré en local sous : {local_output_path}")
