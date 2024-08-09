https://cloud.tencent.com/developer/article/1836076

https://developer.ibm.com/data/project-codenet/#get-this-dataset1

目前总结，terminate容易造成孤儿进程等一系列问题，应当使用event等来让程序正常结束

取数据要一个线程

## 20240808

尽管现在很想给`load_weights`也用并发来实现，甚至实现一个全任务的线程实现，队列始终get，但是会放入到一个全局的字典里，
通信与数据取出采用时间戳或随机token，派发任务时顺便拿到一个之后取数据用的token就行，然后这个token作为key

但是目前还是先实现一个简单版本，之后再做实现，不然遥遥无期。