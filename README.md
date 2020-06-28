# Hierarchical Attention Network with Pairwise Loss for Chinese Zero Pronoun Resolution
Recent neural network methods for Chinese zero pronoun resolution typically model zero pronouns by only utilizing contexts of the zero pronouns while ignoring candidate antecedents information and simply treat the task as a classification task. In this paper, we propose a Hierarchical Attention Network with Pairwise Loss (HAN-PL), for Chinese zero pronoun resolution. In the proposed HAN-PL, we design a two-layer attention model to generate more powerful representations for zero pronouns and candidate antecedents. In addition, we integrate constraint of similarities among correct antecedents into max-margin loss, for guiding the training of the model. Our model achieves state-of-the-art performance on OntoNotes 5.0 dataset.

## Requirements
* Python 2.7
   * Pytorch(0.4.0)
   * CUDA

## Citation

```
@inproceedings{lin2020hierarchical,
  title={Hierarchical Attention Network with Pairwise Loss for Chinese Zero Pronoun Resolution},
  author={Lin, Peiqin and Yang, Meng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={05},
  pages={8352--8359},
  year={2020}
}
```

