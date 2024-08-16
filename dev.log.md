https://cloud.tencent.com/developer/article/1836076

https://developer.ibm.com/data/project-codenet/#get-this-dataset1

目前总结，terminate容易造成孤儿进程等一系列问题，应当使用event等来让程序正常结束

取数据要一个线程

## 20240808

尽管现在很想给`load_weights`也用并发来实现，甚至实现一个全任务的线程实现，队列始终get，但是会放入到一个全局的字典里，
通信与数据取出采用时间戳或随机token，派发任务时顺便拿到一个之后取数据用的token就行，然后这个token作为key

但是目前还是先实现一个简单版本，之后再做实现，不然遥遥无期。

## 20240815
关于当前为什么rload里不能判断workers is None，是因为dataloader.workers被赋值为None是在触发rload之后，
因为rload载入文件比较慢，所以子程序当执行arrange_workers时，都已经是主进程给workers赋值之后了。

## 20240816
基本可以排除是dataloader的问题了。所以还是找不到速度缓慢的原因


## 20240817

1. 为以后更复杂的timer设计做准备，简化体现在llm/trainer.py中
2. ~~初步简化llama3的代码，方便分析训练llama3时为什么显卡占用率这么低~~
2. 测试一下原版llama3的行列并行Linear层的加速效果(*貌似显卡使用率更低了，但是显存占用量显著减少了*)
3. 在150和bs=32时，view和连续内存对速度和显存的优化都不明显（几乎没有）