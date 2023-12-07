# D2TNet
Code of [D2TNet: A ConvLSTM Network with Dual-direction Transfer for Pan-sharpening](https://ieeexplore.ieee.org/abstract/document/9761261)

Tips
---------
#### To train:<br>
* Step1: I suggest you to create your own training dataset by running crop.py, as the training dataset is so large.
* Step2: Run train.py.

#### To test with the pre-trained model:<br>
* In model.py, comment out the 20th line, and run main.py.
  (We also train the network on WorldView-II dataset with 8 channels, and put the pretrained model [here](https://github.com/Meiqi-Gong/PSDNet/tree/main/D2TNet_models_WV2))

If this work is helpful to you, please cite it as:
```
@article{gong2022d2tnet,
  title={D2TNet: A ConvLSTM Network with Dual-direction Transfer for Pan-sharpening},
  author={Gong, Meiqi and Ma, Jiayi and Xu, Han and Tian, Xin and Zhang, Xiao-Ping},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2022},
  publisher={IEEE}
}
```

If you have any question, please email to me (meiqigong@whu.edu.cn).
