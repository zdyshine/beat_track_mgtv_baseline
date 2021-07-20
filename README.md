# 排行榜
当前排行榜为2021年07月19日 12:00:00之前提交成绩  

| 排名 | 参赛队伍 | 分数 | 总耗时（秒） | 最佳成绩提交时间 |
| :----:| :---- | :----: | :----: | :----: |
| 1 | fuqianya | 87.9600 | 50 | 2021/07/20 11:13:12 |
| 2 | 十一月的肖邦 | 87.4800 | 42 | 2021/07/20 10:11:19 |
| 3 | mg13078804b | 87.1400 | 19 | 2021/07/19 10:53:21 |
| 4 | 音乐王子李大猷 | 85.8900 | 133 | 2021/07/20 08:53:31 |
| 5 | DropTheBeat | 85.8800 | 306 | 2021/07/18 13:48:12 |
| 6 | 红鲤鱼与绿鲤鱼与驴 | 85.6664 | 195 | 2021/07/20 11:45:58 |
| 7 | CB | 84.8100 | 519 | 2021/07/20 11:26:42 |
| 8 | 莫扎特儿 | 82.6600 | 324 | 2021/07/20 11:47:22 |
| 9 | 贝多芬 | 82.4600 | 115 | 2021/07/19 09:40:39 |
| 10 | 1024K | 82.2000 | 415 | 2021/07/17 20:53:37 |

说明：  
总耗时（秒）：docker 调起 run.sh脚本开始，到所有预测结束的时间。  
本耗时只作为参考，选手的最终有效成绩以不违背比赛规则的最终审查结果为准。  


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
