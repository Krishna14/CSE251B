This directory contains Pytorch implementation of Dual Attention(DA-RNN) proposed by Qin, Yao, et al in https://arxiv.org/pdf/1704.02971.pdf applied on top of the Sequence to Sequence – Video to Text (S2VT) model as proposed by Venugopalan, Subhashini, et al. in https://arxiv.org/pdf/1505.00487.pdf

The DA-RNN paper proposed a dual attention mechanism for time series prediction. We experiment how this attention mechanism works for video captioning task. 

We use MSVD dataset's video frame features, namely, Resnet101 features. The dataset we used can be taken from this link: https://drive.google.com/drive/folders/1oAAxKCs0ZjrUe7xbE6j-CqWhSubx3F1b?usp=sharing

Please download the data files from the above link inside your project directory. Replace "data_path" variable with this data path, in experiment.py. 
Sample train.ipynb notebook is provided to train the model and evaluate on test dataset. Hyperparameters can be tuned by changing their corresponding variable values in experiment.py and model_factory.py. The evaluation metric used is METEOR score to compare with the referred S2VT paper. 

Implementation References:
1. https://github.com/vijayvee/video-captioning/tree/9dfd6608a520adbd94c97b8e8e8ade9e7c3536b8
2. https://github.com/Peratham/video2text.pytorch/tree/0d530e271ee47ca4aa707dfec8d6c71801277b62
3. Starter Code given in PA4 CSE 251B Winter 2021
4. Final code submitted for PA4 CSE 251B Winter 2021, authored by the same team (Gautham Reddy, Justin Huynh, Sreekrishna Ramaswamy,Tri Truong and Akshaya Raju)
5. https://gist.github.com/vsubhashini/38d087e140854fee4b14
6. https://github.com/Zhenye-Na/DA-RNN
7. http://chandlerzuo.github.io/blog/2017/11/darnn
8. https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
9. https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
10. https://www.nltk.org/_modules/nltk/translate/meteor_score.html



