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

实在是不行了，基本来看，基本不可能有望在单卡上训练了，干脆彻底转向为deepspeed开发吧

显存占用问题先不管了，但是现在终于找到占用率的问题了，通过对前向传播和反向传播的耗时观察来看，每次需要等待才能前向传播的情况下，等待结束多没多久就是
文件的重新载入。也就是queue还是不够大。6， num_workers我居然只设置了1，但是为什么跑的时候好像不是1呢？看起来是有些测试的东西没删干净导致的。还是
不对啊，到底是为什么啊？