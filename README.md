# COMETA: the corpus of online medical entities

This repo holds code for running baseline models presented in our [paper](https://www.aclweb.org/anthology/2020.emnlp-main.253/): *COMETA: A Corpus for Medical Entity Linking in the Social Media* at EMNLP 2020.

COMETA is an entity linking dataset of layman medical terminology. It has been collected by analysing four years of content in 68 health-themed subreddits and annotating the most frequent with their corresponding SNOMED-CT entities. Each term is assigned two annotations: a General SNOMED-CT identifier and a Specific one, denoting respectively the literal and contextual meaning of the term. 

For a copy of the corpus, please download [here](https://www.dropbox.com/s/e8mdpberw959xhj/cometa.zip?dl=0).

## Pretrained vectors

| Model      | Download Link                                                                                                                                                                                 |
|-----------|----------------|
| Bioreddit-FastText  | [bin](https://drive.google.com/file/d/14xWIMx90VAxpjhPtCF_ECQemAMuSmvfU/view?usp=sharing), [vec](https://drive.google.com/file/d/1CTZEO9pvR3C8DbxJ7bt-y4t3TWGirQ_0/view?usp=sharing)                                                                                                                                                                                               |
| Bioreddit-BERT      | [huggingface](https://huggingface.co/cambridgeltl/BioRedditBERT-uncased)                                                                                                                                                                                                |

You can find vectors trained on the same Bioreddit corpus for ELMo, Flair and GloVE in [this repository](https://github.com/basaldella/bioreddit).

### Citation

If you use our corpus or our embeddings, please cite:

```bibtex
@inproceedings{basaldella-etal-2020-cometa,
    title = "{COMETA}: A Corpus for Medical Entity Linking in the Social Media",
    author = "Basaldella, Marco  and
      Liu, Fangyu  and
      Shareghi, Ehsan  and
      Collier, Nigel",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.253",
    pages = "3122--3137",
}
```
