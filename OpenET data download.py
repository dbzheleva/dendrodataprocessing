#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 14:57:39 2025

@author: dragomirazheleva
"""
import geopandas as gpd

import openet_client                     # <── CHANGE HERE
from pathlib import Path
import time




import json, pandas as pd, time
import openet_client

import importlib.metadata as im
print(im.version("openet-client")) 

        # pip install openet-client

API_KEY = "IYruPiEyB6NBGjzXrb18NagSYGAECKHhZchY7uO8lZQ1nQMdZv7TC239JAPj"
openet  = openet_client.OpenETClient(API_KEY)

print([m for m in dir(openet.raster.timeseries.raster_manager.timeseries) if not m.startswith('_')])
# should list:  ['__call__', '__doc__', ..., 'area', 'multipolygon', ...]      
ts2 = openet.raster.timeseries.raster_manager.timeseries
print([m for m in dir(ts2) if not m.startswith('_')])

#%%

# --- load your polygons -------------------------------------------------
gjs = json.load(open("/Users/dragomirazheleva/Downloads/map-2.geojson"))      # 16 features
import json, requests, pandas as pd, time, zipfile, io

API_KEY = "IYruPiEyB6NBGjzXrb18NagSYGAECKHhZchY7uO8lZQ1nQMdZv7TC239JAPj"
GEOM_FC  = json.load(open("/Users/dragomirazheleva/Downloads/map-2.geojson"))      # your FeatureCollection

body = {
    "geometry":   GEOM_FC,
    "date_range": ["2023-01-01", "2023-12-31"],
    "interval":   "monthly",        # daily | monthly | annual
    "model":      "ensemble",       # or "ptjpl", "ssebop", ...
    "units":      "mm"
}

headers = {
    "accept":       "application/json",
    "Content-Type": "application/json",
    "Authorization": API_KEY
}

url = "https://openet-api.org/raster/timeseries/multipolygon"
print("Posting…")
resp = requests.post(url, json=body, headers=headers, timeout=90)
resp.raise_for_status()

# the API replies with a URL pointing to a zipped CSV in Earth Engine's bucket
download_url = resp.json()["url"]
print("Download URL:", download_url)

print("Fetching zipped CSV…")
csv_zip = requests.get(download_url, timeout=120).content
zf = zipfile.ZipFile(io.BytesIO(csv_zip))
csv_name = zf.namelist()[0]
df = pd.read_csv(zf.open(csv_name))

df.to_csv("Orchards_ET_2023.csv", index=False)
print("✓ wrote", df.shape[0], "rows to Orchards_ET_2023.csv")
