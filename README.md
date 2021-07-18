# 排行榜
当前排行榜为2021年07月18日 12:00:00之前提交成绩  

| 排名 | 参赛队伍 | 分数 | 总耗时（秒） |
| :----:| :---- | :----: | :----: |
| 1 | mg13078804b | 86.0800 | 24 |
| 2 | DropTheBeat | 85.3500 | 518 |
| 3 | 十一月的肖邦 | 84.4700 | 41 |
| 4 | 音乐王子李大猷 | 84.4300 | 166 |
| 5 | 1024K | 82.2000 | 415 |
| 6 | 伯牙子期 | 82.1100 | 326 |
| 7 | 红鲤鱼与绿鲤鱼与驴 | 81.8100 | 125 |
| 8 | fuqianya | 81.7600 | 85 |
| 9 | CB | 81.1100 | 932 |
| 10 | mg2353752Iy | 80.1500 | 730 |

# beat_track_mgtv_baseline

An implementation of two adaptations to Davies &amp; Böck's beat-tracking temporal convolutional network [1].

## Usage

Run:
```
1. 修改config.yaml的文件路径,指定为自己的数据  
2. 运行Step1_dataprocess.py,得到处理好的数据
3. 运行Step2_train.py,开始进行训练
4. 运行Step3_beat_tracker.py,得到beat标注文件
####################
5. run.py提交样例代码，选手根据自己的情况进行修改
####################
```

## References

[1] M. E. P. Davies and S. Bock, _‘Temporal convolutional networks for musical audio beat tracking’_, in 2019 27th European Signal Processing Conference (EUSIPCO), A Coruna, Spain, 2019, pp. 1–5, doi: 10.23919/EUSIPCO.2019.8902578.    
[2] https://github.com/ldzhangyx/TCN-for-beat-tracking  

## 其他  

本代码只提供了beat的解题思路，beat线上分数是8.8分，优化空间还很大。  
优化方向： 
```
1.数据处理  
2.网络改进  
3.设计方案优化downbeat
```
希望各位选手都能取得好成绩。
