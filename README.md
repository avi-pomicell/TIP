# Tri-graph Information Propagation (TIP) model

TIP is an efficient general approach for **multi-relational link prediction** in any **multi-modal**  (i.e. heterogeneous and multi-relational) network with two types of nodes. It can also be applied to the **Knowledge Graph Completion** and **Recommendation** task. TIP model is inspired by the [Decagon](https://github.com/marinkaz/decagon) and [R-GCN](https://github.com/tkipf/relational-gcn) models, motivated by their limitations of high computational cost and memory demand when graph goes really complex. TIP improves their link prediction **accuracy**, and time and space **efficiency** of node representation learning. See details on the algorithm in our paper [(Xu, Sang, and Lu, 2019)](https://grlearning.github.io/papers/94.pdf).

## TIP for Polypharmacy Side Effect Prediction

we are particularly concerned about the safety of [polypharmacy](https://en.wikipedia.org/wiki/Polypharmacy), which is the concurrent use of multiple medications by a patient. Given a pair of drugs (:pill:,:pill:), the TIP model will predict how many polypharmacy side effects the drug pair will have, and what are the possibilities.

<div align=center>
<img height="100" src="img/pred_dd.png" alt=""hhh/>
</div>

We use *POSE clinical records* and *pharmacological information* to construct a multi-modal biomedical graph with two types of nodes: Drug (D) and Protein (P). The graph contains three types of interaction (refer to three subgraphs): 

&emsp; :cookie: &ensp; D-D graph: drug-drug interactions with side effects as edge labels

&emsp; :cake: &ensp; P-D graph: protein-drug interactions (with a fixed label)

&emsp; :ice_cream: &ensp; P-P graph: protein-protein interactions (with a fixed label)

<div align=center>
<img width="500" src="img/network.png" alt=""hhh/>
</div>

TIP model embeds proteins and drugs into different spaces of possibly different dimensions in the encoder, and predict side effects of drug combinations in the decoder. As shown below, TIP learns the protein embedding firstly on the P-P graph, and passes it to D-D graph via D-P graph. On D-D graph, TIP learns drug embedding and predicts relationships between drugs.

**TIP Encoder**:

<div align=center>
<img height="300" src="img/encoder.png">
</div>

**TIP Decoder**:

<div align=center>
<img height="300" src="img/decoder.png">
</div>

## Source Code

TIP is implemented in [PyTorch]([`pytorch`](https://pytorch.org/)) with [PyG](https://github.com/rusty1s/pytorch_geometric) package. It is developed and tested under Python 3.  

### Requirement

You can install the `pytorch` and `pyg` packages with the versions that matches your hardware, or use the same environment as mine using the following commands:

```shell
conda create -y -n tip-gpu python==3.9
conda install -y -n tip-gpu pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch 
conda install -y -n tip-gpu pyg==2.0.1 -c pyg -c conda-forge	
```

*(Optional)* If you are interested in monitoring GPU memory usage of the model, the `pytorch_memlab` package is helpful.
```shell
pip install pytorch_memlab
```

*(Optional)* TIP is trained and tested on a single **GPU**. If you are interested in training TIP using multiple GPUs, `pytorch_lightning` would be helpful.

### Running

The processed data and the code for data processing are in the `./data/` folder. The raw datasets are available on the [BioSNAP](http://snap.stanford.edu/biodata/index.html). See `./data.ipynb` for the full polypharmacy datasets analysis and data preprocessing.

**Step 1**: preparing data. Run it once to generate a `data_dict.pkl` file in `./data/` folder).
```shell
python prepare.py			
```

**Step 2**: training and testing model. The default model is `TIP-cat`. If you want to train and test a `TIP-add` model, change the value of variable `MOD` from `'cat'` to `'add'`. 
```shell
python tip.py
```

By following the above steps and using the default hyper-parameter settings, the results that are shown in the TIP paper [(Xu, Sang, and Lu, 2019)](https://grlearning.github.io/papers/94.pdf) can be reproduced.

:new_moon_with_face::waxing_crescent_moon::first_quarter_moon::waxing_gibbous_moon: **Please browse/open issues should you have any questions or ideas**​ :waning_gibbous_moon::last_quarter_moon::waning_crescent_moon::new_moon_with_face:

## Cite Us
If you found this work useful, please cite us:
```
@article{xu2019tip,
	title={Tri-graph Information Propagation for Polypharmacy Side Effect Prediction},
	author={Hao Xu and Shengqi Sang and Haiping Lu},
	journal={NeurIPS Workshop on Graph Representation Learning},
	year={2019}
}
```

## License

TIP is licensed under the MIT License.

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/sangsq"><img src="https://avatars.githubusercontent.com/u/16742808?v=4?s=100" width="100px;" alt=""/><br /><sub><b>sangsq</b></sub></a><br /><a href="https://github.com/NYXFLOWER/TIP/commits?author=sangsq" title="Code">💻</a> <a href="https://github.com/NYXFLOWER/TIP/commits?author=sangsq" title="Tests">⚠️</a> <a href="#ideas-sangsq" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/NYXFLOWER/TIP/commits?author=sangsq" title="Documentation">📖</a></td>
    <td align="center"><a href="https://haipinglu.github.io/"><img src="https://avatars.githubusercontent.com/u/23463961?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Haiping Lu</b></sub></a><br /><a href="https://github.com/NYXFLOWER/TIP/commits?author=haipinglu" title="Documentation">📖</a> <a href="#a11y-haipinglu" title="Accessibility">️️️️♿️</a></td>
    <td align="center"><a href="https://github.com/shree970"><img src="https://avatars.githubusercontent.com/u/41207097?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Shreeyash</b></sub></a><br /><a href="https://github.com/NYXFLOWER/TIP/issues?q=author%3Ashree970" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/Chertoganov"><img src="https://avatars.githubusercontent.com/u/40623363?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Chertoganov</b></sub></a><br /><a href="https://github.com/NYXFLOWER/TIP/issues?q=author%3AChertoganov" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://www.jianshu.com/u/31c221f09d8a"><img src="https://avatars.githubusercontent.com/u/25343084?v=4?s=100" width="100px;" alt=""/><br /><sub><b>ZillaRU</b></sub></a><br /><a href="https://github.com/NYXFLOWER/TIP/issues?q=author%3AZillaRU" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/gilgamsh"><img src="https://avatars.githubusercontent.com/u/56181610?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jiaxi Jiang  </b></sub></a><br /><a href="https://github.com/NYXFLOWER/TIP/issues?q=author%3Agilgamsh" title="Bug reports">🐛</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
