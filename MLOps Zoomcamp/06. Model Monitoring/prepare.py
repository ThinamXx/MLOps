# IMPORTING MODULES:
import ast
from tqdm import tqdm
import requests



# DOWNLOADING THE DATASETS:
files = [("green_tripdata_2021-01.parquet", "."), ("green_tripdata_2021-01.parquet", "./evidently_service/datasets")]
print("Downloading the dataset")
for file, path in files:
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{file}"
    resp = requests.get(url, stream=True)
    save_path = f"{path}/{file}"
    with open(save_path, "wb") as handle:
        for data in tqdm(
                resp.iter_content(),
                desc=f"{file}",
                postfix=f"save to {save_path}",
                total=int(resp.headers["Content-Length"])):
            handle.write(data)
