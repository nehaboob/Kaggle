#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 12:50:13 2017

@author: nehaboob
"""
# Usage: fix_keras_model.py old_model.h5 new_model.h5


import h5py
import shutil
import json
import sys

input_model_path = sys.argv[1]
output_model_path = sys.argv[2]
shutil.copyfile(input_model_path, output_model_path)

with h5py.File(output_model_path, "r+") as out_h5:
  v = out_h5.attrs.get("model_config")
  config = json.loads(v)
  for i, l in enumerate(config["config"]["layers"]):
    dtype = l["config"].pop("input_dtype", None)
    if dtype is not None:
      l["config"]["dtype"] = dtype

  new_config_str = json.dumps(config)
  out_h5.attrs.modify("model_config", new_config_str)

# Check that it worked.
from keras.models import load_model
load_model(output_model_path)