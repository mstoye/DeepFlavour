This reads the input and adds nice function to the dataset:
ReadDeepBs.py
In there you can add the pathes to you files. The train sample will be used for training. validation and test or for testing. For validation and test sample the estimated labels will be stored for quick checks.


This runs the code and sets hyperparameters, it also store the estimated truth for a validation and test sample:
python DeepB_KERAS.py

The output will be :
Architecture_cMVA_v6.json : stores the NN architecture and can be read by LWNN in C++.
DeeBFlovour_KERAS_cMVA_Debug_v6.h5: are the wights that can be read by  LWNN
KERAS_test_result_v6.npy and KERAS_val_result_v6.npy: are the estimated truth label probabilities per event (jet)

if you run on CMG (CERN) GPU you have to first
