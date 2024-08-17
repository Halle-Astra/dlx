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

现在这一组参数，显存能接受，而且推理要快一点点，135M的参数规模，很迷。

    args = {
        "dim": 512,
        "n_layers": 1,
        "n_heads": 2,
        "n_kv_heads": 2,
        "vocab_size": 128256,
        "multiple_of": 1024,
        "ffn_dim_multiplier": 1.3,
        "norm_eps": 1e-05,
        "rope_theta": 500000.0,
        "max_seq_len": 150,
        "mode": "train"
    }
    margs = ModelArgs(**args)

n_layers改成2都能运行。离谱.

## 20240818

现在开始专门的时间测速了，主要目的是验证是不是mask的生成导致的时间占用。

参数：
    
    args = {
        "dim": 512,
        "n_layers": 8,
        "n_heads": 1,
        "n_kv_heads": 1,
        "vocab_size": 128256,
        "multiple_of": 1024,
        "ffn_dim_multiplier": 1.3,
        "norm_eps": 1e-05,
        "rope_theta": 500000.0,
        "max_seq_len": 150,
        "mode": "train"
    }

耗时分析如下：（总之问题还是不在这）

    2024-08-17 16:30:33.446 | INFO     | dlx.utils.stat:stat_parameters_num:24 - trainable params: 164.897M
    2024-08-17 16:30:33.447 | INFO     | dlx.utils.stat:stat_parameters_num:25 - untrainable params: 0.000K
    2024-08-17 16:30:33.449 | INFO     | dlx.utils.stat:stat_parameters_num:26 - total params: 164.897M
    2024-08-17 16:30:35.641 | DEBUG    | dlx.train.llm.trainer:start:166 - 0, cost of catching batch: 0.0032465457916259766s
    2024-08-17 16:30:35.820 | DEBUG    | dlx.models.llm.llama3:forward:369 - time of generate mask: 0.08592987060546875
    2024-08-17 16:30:41.626 | DEBUG    | dlx.train.llm.trainer:start:177 - cost of forward :5.985889673233032
    2024-08-17 16:30:50.039 | DEBUG    | dlx.train.llm.trainer:start:189 - cost of backward: 8.41292119026184
    2024-08-17 16:30:50.045 | INFO     | dlx.train.llm.trainer:log_training:137 - step: 0 | loss: 11.928168296813965 | max waiting batch: 0.003s
    2024-08-17 16:30:50.056 | DEBUG    | dlx.train.llm.trainer:start:166 - 1, cost of catching batch: 0.010938167572021484s
    2024-08-17 16:30:50.111 | DEBUG    | dlx.models.llm.llama3:forward:369 - time of generate mask: 0.027195215225219727
    2024-08-17 16:30:52.602 | DEBUG    | dlx.train.llm.trainer:start:177 - cost of forward :2.545494556427002
    2024-08-17 16:30:57.480 | DEBUG    | dlx.train.llm.trainer:start:189 - cost of backward: 4.878072023391724
    2024-08-17 16:30:57.491 | DEBUG    | dlx.train.llm.trainer:start:166 - 2, cost of catching batch: 0.010713577270507812s
    2024-08-17 16:30:57.545 | DEBUG    | dlx.models.llm.llama3:forward:369 - time of generate mask: 0.026389360427856445
    2024-08-17 16:31:00.077 | DEBUG    | dlx.train.llm.trainer:start:177 - cost of forward :2.5858421325683594
    2024-08-17 16:31:05.031 | DEBUG    | dlx.train.llm.trainer:start:189 - cost of backward: 4.954129457473755
    2024-08-17 16:31:05.048 | DEBUG    | dlx.train.llm.trainer:start:166 - 3, cost of catching batch: 0.01584458351135254s

继续分析是不是DDP在每次backward之后都要多卡同步一下导致的。


还是不行，也不是DDP的问题，可能是llama3的复数ROPE实现导致的吧，真是迷惑，下次用我自己实现的Transformer来做吧，不知道了。

    2024-08-17 16:58:18.485 | INFO     | dlx.utils.stat:stat_parameters_num:24 - trainable params: 164.897M
    2024-08-17 16:58:18.497 | INFO     | dlx.utils.stat:stat_parameters_num:25 - untrainable params: 0.000K
    2024-08-17 16:58:18.502 | INFO     | dlx.utils.stat:stat_parameters_num:26 - total params: 164.897M
    2024-08-17 16:58:20.404 | DEBUG    | dlx.train.llm.trainer:start:175 - 0, cost of catching batch: 0.008106231689453125s
    2024-08-17 16:58:20.447 | DEBUG    | dlx.models.llm.llama3:forward:369 - time of generate mask: 0.030376434326171875
    2024-08-17 16:58:27.719 | DEBUG    | dlx.train.llm.trainer:start:186 - cost of forward :7.315085172653198
    2024-08-17 16:58:34.419 | DEBUG    | dlx.train.llm.trainer:start:198 - cost of backward: 6.699707746505737
    2024-08-17 16:58:34.430 | INFO     | dlx.train.llm.trainer:log_training:146 - step: 0 | loss: 11.941049575805664 | max waiting batch: 0.008s
    2024-08-17 16:58:34.441 | DEBUG    | dlx.train.llm.trainer:start:175 - 1, cost of catching batch: 0.010625123977661133s
    2024-08-17 16:58:34.473 | DEBUG    | dlx.models.llm.llama3:forward:369 - time of generate mask: 0.026329517364501953
    2024-08-17 16:58:37.000 | DEBUG    | dlx.train.llm.trainer:start:186 - cost of forward :2.5587728023529053
    2024-08-17 16:58:42.003 | DEBUG    | dlx.train.llm.trainer:start:198 - cost of backward: 5.003100395202637
    2024-08-17 16:58:42.020 | DEBUG    | dlx.train.llm.trainer:start:175 - 2, cost of catching batch: 0.010692834854125977s
    2024-08-17 16:58:42.052 | DEBUG    | dlx.models.llm.llama3:forward:369 - time of generate mask: 0.02658534049987793
    2024-08-17 16:58:44.697 | DEBUG    | dlx.train.llm.trainer:start:186 - cost of forward :2.676975727081299
    2024-08-17 16:58:49.781 | DEBUG    | dlx.train.llm.trainer:start:198 - cost of backward: 5.0847249031066895
    2024-08-17 16:58:49.794 | DEBUG    | dlx.train.llm.trainer:start:175 - 3, cost of catching batch: 0.011670589447021484s
    2024-08-17 16:58:49.837 | DEBUG    | dlx.models.llm.llama3:forward:369 - time of generate mask: 0.03143954277038574
    2024-08-17 16:58:52.480 | DEBUG    | dlx.train.llm.trainer:start:186 - cost of forward :2.6865458488464355
    2024-08-17 16:58:57.473 | DEBUG    | dlx.train.llm.trainer:start:198 - cost of backward: 4.9930174350738525


服了，真服了，也不是这个原因，直接删了加rope的代码，然后还是没区别

    2024-08-17 17:07:09.590 | INFO     | dlx.utils.stat:stat_parameters_num:24 - trainable params: 164.897M
    2024-08-17 17:07:09.591 | INFO     | dlx.utils.stat:stat_parameters_num:25 - untrainable params: 0.000K
    2024-08-17 17:07:09.592 | INFO     | dlx.utils.stat:stat_parameters_num:26 - total params: 164.897M
    2024-08-17 17:07:11.397 | DEBUG    | dlx.train.llm.trainer:start:175 - 0, cost of catching batch: 0.00875711441040039s
    2024-08-17 17:07:11.456 | DEBUG    | dlx.models.llm.llama3:forward:368 - time of generate mask: 0.010188102722167969
    2024-08-17 17:07:15.569 | DEBUG    | dlx.train.llm.trainer:start:186 - cost of forward :4.171674013137817
    2024-08-17 17:07:26.454 | DEBUG    | dlx.train.llm.trainer:start:198 - cost of backward: 10.885590553283691
    2024-08-17 17:07:26.460 | INFO     | dlx.train.llm.trainer:log_training:146 - step: 0 | loss: 11.937785148620605 | max waiting batch: 0.009s
    2024-08-17 17:07:26.478 | DEBUG    | dlx.train.llm.trainer:start:175 - 1, cost of catching batch: 0.012222766876220703s
    2024-08-17 17:07:26.516 | DEBUG    | dlx.models.llm.llama3:forward:368 - time of generate mask: 0.026329755783081055
    2024-08-17 17:07:28.590 | DEBUG    | dlx.train.llm.trainer:start:186 - cost of forward :2.112217903137207
    2024-08-17 17:07:33.503 | DEBUG    | dlx.train.llm.trainer:start:198 - cost of backward: 4.91229772567749
    2024-08-17 17:07:33.519 | DEBUG    | dlx.train.llm.trainer:start:175 - 2, cost of catching batch: 0.015867948532104492s
    2024-08-17 17:07:33.562 | DEBUG    | dlx.models.llm.llama3:forward:368 - time of generate mask: 0.031778573989868164
    2024-08-17 17:07:35.628 | DEBUG    | dlx.train.llm.trainer:start:186 - cost of forward :2.109123468399048
    2024-08-17 17:07:40.450 | DEBUG    | dlx.train.llm.trainer:start:198 - cost of backward: 4.821685314178467
    2024-08-17 17:07:40.472 | DEBUG    | dlx.train.llm.trainer:start:175 - 3, cost of catching batch: 0.010674476623535156s
    2024-08-17 17:07:40.509 | DEBUG    | dlx.models.llm.llama3:forward:368 - time of generate mask: 0.02675938606262207
    2024-08-17 17:07:42.613 | DEBUG    | dlx.train.llm.trainer:start:186 - cost of forward :2.1410765647888184
    2024-08-17 17:07:47.453 | DEBUG    | dlx.train.llm.trainer:start:198 - cost of backward: 4.8405561447143555
    2024-08-17 17:07:47.482 | DEBUG    | dlx.train.llm.trainer:start:175 - 4, cost of catching batch: 0.023192882537841797s
    2024-08-17 17:07:47.520 | DEBUG    | dlx.models.llm.llama3:forward:368 - time of generate mask: 0.021424531936645508
    2024-08-17 17:07:49.535 | DEBUG    | dlx.train.llm.trainer:start:186 - cost of forward :2.0527472496032715

关于backward阶段的耗时：
    
    2024-08-17 17:18:15.062 | DEBUG    | dlx.models.llm.llama3:forward:369 - time of generate mask: 0.026854276657104492
    2024-08-17 17:18:17.703 | DEBUG    | dlx.train.llm.trainer:start:191 - cost of forward :2.689047336578369
    2024-08-17 17:18:19.216 | DEBUG    | dlx.train.llm.trainer:_backward:160 - time of grad cal: 1.0575227737426758
    2024-08-17 17:18:22.624 | DEBUG    | dlx.train.llm.trainer:_backward:165 - time of optim: 3.4075379371643066
    2024-08-17 17:18:22.624 | DEBUG    | dlx.train.llm.trainer:start:203 - cost of backward: 4.921015024185181
    2024-08-17 17:18:22.641 | DEBUG    | dlx.train.llm.trainer:start:180 - 27, cost of catching batch: 0.010694742202758789s
    2024-08-17 17:18:22.676 | DEBUG    | dlx.models.llm.llama3:forward:369 - time of generate mask: 0.03471970558166504
    2024-08-17 17:18:25.195 | DEBUG    | dlx.train.llm.trainer:start:191 - cost of forward :2.5543599128723145
    2024-08-17 17:18:26.771 | DEBUG    | dlx.train.llm.trainer:_backward:160 - time of grad cal: 1.0742831230163574
    2024-08-17 17:18:30.272 | DEBUG    | dlx.train.llm.trainer:_backward:165 - time of optim: 3.500648021697998
    2024-08-17 17:18:30.272 | DEBUG    | dlx.train.llm.trainer:start:203 - cost of backward: 5.077411651611328
    2024-08-17 17:18:30.284 | DEBUG    | dlx.train.llm.trainer:start:180 - 28, cost of catching batch: 0.010411262512207031s
    2024-08-17 17:18:30.322 | DEBUG    | dlx.models.llm.llama3:forward:369 - time of generate mask: 0.026834487915039062
    2024-08-17 17:18:32.790 | DEBUG    | dlx.train.llm.trainer:start:191 - cost of forward :2.5066637992858887
    2024-08-17 17:18:34.406 | DEBUG    | dlx.train.llm.trainer:_backward:160 - time of grad cal: 1.1225650310516357
    2024-08-17 17:18:37.812 | DEBUG    | dlx.train.llm.trainer:_backward:165 - time of optim: 3.4059510231018066
    2024-08-17 17:18:37.818 | DEBUG    | dlx.train.llm.trainer:start:203 - cost of backward: 5.027588367462158
    2024-08-17 17:18:37.829 | DEBUG    | dlx.train.llm.trainer:start:180 - 29, cost of catching batch: 0.010773181915283203s
    2024-08-17 17:18:37.872 | DEBUG    | dlx.models.llm.llama3:forward:369 - time of generate mask: 0.037316083908081055
    2024-08-17 17:18:40.318 | DEBUG    | dlx.train.llm.trainer:start:191 - cost of forward :2.4893863201141357
    2024-08-17 17:18:41.944 | DEBUG    | dlx.train.llm.trainer:_backward:160 - time of grad cal: 1.125314474105835
    2024-08-17 17:18:45.370 | DEBUG    | dlx.train.llm.trainer:_backward:165 - time of optim: 3.4260201454162598
    2024-08-17 17:18:45.371 | DEBUG    | dlx.train.llm.trainer:start:203 - cost of backward: 5.052683591842651
    2024-08-17 17:18:45.382 | DEBUG    | dlx.train.llm.trainer:start:180 - 30, cost of catching batch: 0.010607719421386719s
    2024-08-17 17:18:45.420 | DEBUG    | dlx.models.llm.llama3:forward:369 - time of generate mask: 0.02653646469116211
    2024-08-17 17:18:47.987 | DEBUG    | dlx.train.llm.trainer:start:191 - cost of forward :2.6052064895629883
    


有没有一种可能，是因为loss_module因为继承自nn.Module所以必须要加上.to(device)呢？(**也不对！**)

