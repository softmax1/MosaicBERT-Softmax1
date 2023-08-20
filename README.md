# MosaicBERT-Softmax1

A test of the [Attention Is Off By One](https://www.evanmiller.org/attention-is-off-by-one.html) hypothesis.
MosaicML claims that with their recipe you can [pretrain BERT from scratch for $20](https://www.mosaicml.com/blog/mosaicbert).
As such, I will test the hypothesis by generalizing their implementation of [BERT](https://github.com/mosaicml/examples/tree/main/examples/benchmarks/bert) to use my implementation of [FlashAttention with Softmax_n](https://github.com/softmax1/FlashAttention-with-Softmax1).

## Training Dataset
I train on the [Colossal, Cleaned, Common Crawl](https://huggingface.co/datasets/c4) (C4) dataset.
Mosaic used 78.6% of the 'en' subset of C4 for their pretraining.
Note that MosaicBERT reached the performance level of the original BERT's average GLUE score (of 79.6) in 21.4% of its total training time.
Therefore, to save resources, I will use 16.8% of the 'en' subset of C4 for training, which corresponds to 61,358,080 samples or ~135 GB.
The main reason IMO why outliers did not appear in my [previous test](https://github.com/softmax1/EsperBERTo/tree/main) of the hypothesis was that the dataset I used was only 178 MB, which was too small.
Using a significantly larger dataset eliminates this possibility.