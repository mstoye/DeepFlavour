This reads the input and adds nice function to the dataset:
ReadDeepBs.py

This runs the code and sets hyperparameters, it also store the estimated truth for a validation and test sample:
DeepB_with_summaries.py

To run, put the right paths into ReadDeepBs.py and than do:

python DeepB_with_summaries.py

That's it. "with_summaries" means that you can run tensorboard afterwards to see some plots on training ect., i.e. summary files are written.
