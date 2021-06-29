# beat_track_mgtv_baseline

An implementation of two adaptations to Davies &amp; Böck's beat-tracking temporal convolutional network [1].

## Usage

Train:
```
1. 修改config.yaml的文件路径,指定为自己的数据  
2. 运行Step1_dataprocess.py,得到处理好的数据
3. 运行Step2_train.py,开始进行训练
4. 运行Step3_beat_tracker.py,得到beat标注文件
```

## References

[1] M. E. P. Davies and S. Bock, _‘Temporal convolutional networks for musical audio beat tracking’_, in 2019 27th European Signal Processing Conference (EUSIPCO), A Coruna, Spain, 2019, pp. 1–5, doi: 10.23919/EUSIPCO.2019.8902578.    
[2] https://github.com/ldzhangyx/TCN-for-beat-tracking  
