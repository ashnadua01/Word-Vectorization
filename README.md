## Assignment 3 - INLP
### Ashna Dua, 2021101072

### Execution of files:
1. svd.py:
    - Three tags have been implemented --e for Embeddings Size and --w for Window Size and --s for save flag.
    - Any other tag input would lead to an Error

Other functions, to train and evaluate the models are implemented in their respective files.

2. svd-classification.py:
    - Five tags have been implemented --e for Embeddings Size and --w for Window Size and --s for save flag for embeddings --use_pretrained for using pre-trained embeddings, and --save_model for saving the trained model.
    - Any other tag input would lead to an Error

1. skip_gram.py:
    - Three tags have been implemented --e for Embeddings Size and --w for Window Size and --s for save flag.
    - Any other tag input would lead to an Error

Other functions, to train and evaluate the models are implemented in their respective files.

2. skip-gram-classification.py:
    - Five tags have been implemented --e for Embeddings Size and --w for Window Size and --s for save flag for embeddings --use_pretrained for using pre-trained embeddings, and --save_model for saving the trained model.
    - Any other tag input would lead to an Error

- Models folder contains all the pre-trained embeddings and models
- Checkpoints contains the hyper parameter trained models.


Implementation and Results are explained in the report.
One of the assumptions, is that torch behaves unexpectedly with my pre-trained embeddings. The same embeddings give a very high accuracy in some cases, wheras in other cases, it drops to 50-55% as well. This happens is Kaggle, as well as Colab, and only for pre-trained embeddings. If these embeddings are trained, and used to train the LSTM in the same environment, it would result in high accuracy.

My pre-trained models, are facing some error, and display different accuracies (very large difference).

all checkpoints available at:  https://drive.google.com/file/d/1PIMRKtfcEKg-iYIq3ilK6M4S5HWMdkR1/view?usp=sharing
all models available at: https://drive.google.com/file/d/1GTKQFdGryBaEYBgFSjKAo6cFQ1kWx-ZI/view?usp=sharing