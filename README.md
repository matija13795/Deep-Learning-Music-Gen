# Deep-Learning-Music-Gen

There are two folders within this directory:
- OLD
- NEW

Within OLD is a FCNN and Vanilla RNN built on a small Nottingham dataset of 1034 songs. We had some success with the FCNN. However, when we tried to train the Vanilla RNN, we didn't see much of an improvement. We thought this was because we didn't have enough data. Thus, we looked for a new dataset.

Within NEW is where the majority of our work is focused on. We built an LSTM and GRU built on a subset of the large IrishMAN dataset (34, 211 songs). We didn't retrain the FCNN and Vanilla RNN from earlier as we would have to adapt those pipelines to the new tokenizer we built for this dataset and we didn't have time for that. To generate music with our models, use the sample.ipynb notebook.

Follow inline comments for more description on our implementation