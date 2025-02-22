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

为了排除是pytorch版本问题，现在搞个新的容器

    docker run -itd \
            --gpus 1 \
            --name dlx \
            -v /home/halle/data/programs/LLMs:/workspace \
            -v /media/halle/GTA/data/llms/WuDaoCorpus2.0_base_200G:/dataset/fd5061f6/chinese_data/WuDao \
            --net host \
            ubuntu

试过了，也不是，麻了，只能对比对比别人的了，找不到原因反正。要不就onnx导出分析分析。

终于找到原因了，就是dataloader的多线程里的while循环太容易抢到解释器锁了，然后就while循环太多了！加了time.sleep(3)给两个线程里的while部分就
放开解释器给训练逻辑了。

下面这一段，是将单精度训练下的利用率从（60-90）浮动优化为（98-99）浮动的关键（半精度则是96-97）：（目前实践下来，if和logger有点影响，不算大，大的重点还是在dataloader的多线程占用
但是如果将下面的400改为4，就会发生严重的等待，而导致利用率又非常低。在没有4也没有400的情况下，去掉训练中的if和logger会有速度，可以将60的最低值提升到
70甚至80，但是这不排除是因为有机会抢到解释器锁的原因，因为每次对loss.item()或其他and之类的判断，可能会因为内存的搬挪和分配host内存给临时变量
而导致放手解释器锁，目前是这样子的猜测。因此核心还是在dataloader的问题上。）

    while not dataloader_instance.watcher_exit_event.is_set():
        if dataloader_instance.debug:
            if not dataloader_instance.data_queue.empty():
                dataloader_instance.data_queue.get()
        else:
            if not dataloader_instance.length>400 and not dataloader_instance.data_queue.empty():
                batch = []
                for i in range(dataloader_instance.batch_size):
                    sample = dataloader_instance.data_queue.get()
                    batch.append(sample)


## 20240817

目前测试下面这种利用Event来进行线程等待的，但是实测最高显卡利用率只能在91，上限不如sleep(3)的方案。

    def default_generate_batch(dataloader_instance, collate_fn=None):
        while not dataloader_instance.watcher_exit_event.is_set():
            if dataloader_instance.debug:
                if not dataloader_instance.data_queue.empty():
                    dataloader_instance.data_queue.get()
            else:
                if True:
                    #logger.debug('waiting event')
                    dataloader_instance.need_data_event.wait()
                    #logger.debug('end waiting')
                #if not dataloader_instance.length>400 and not dataloader_instance.data_queue.empty():
                    batch = []
                    for i in range(dataloader_instance.batch_size):
                        sample = dataloader_instance.data_queue.get()
                        batch.append(sample)
                    if collate_fn is not None:
                        batch = collate_fn(batch)
                        if (batch is None) or (batch == []):
                            continue
                    dataloader_instance.data_list.append(batch)
                    dataloader_instance.length += 1
                    if dataloader_instance.length >= wait_num:
                        dataloader_instance.need_data_event.clear()
                else:  # Can sleep few time since the data_queue is empty, that's no matter to sleep.
                    time.sleep(3)

总之，目前先到这吧，优化方案也有了（统计3s需要的batch数，然后乘上系数作为条件），先到这了。


# 20250126

关于我的双卡机器的温度和噪声控制问题，目前比较好的控制组合是

功率墙： 150w

风扇控制命令为： `sudo $(which coolgpus)  --temp 50   88 90  --speed 27 40  50`

这样之后，稳定的状态为
```
Fan  Temp  Perf  Pwr:Usage/Cap│         Memory-Usage │ GPU-Util  Compute M. 
40%   86C    P2   134W / 150W │  17795MiB / 22.00GiB │     98%      Default
```

相比满功耗，dlx的训练时间，从92小时增加至117小时，比预期要好很多。

# 20250208

由于训练总是在一小时内出现torch计算速度极其缓慢的情况，现在开始进行问题排查。

torch相关的原有版本：

``` 
nvidia-dlprof-pytorch-nvtx    1.0.0
pytorch-quantization          2.1.0
pytorch-transformers          1.1.0
torch                         1.9.0a0+df837d0
torchtext                     0.9.0a0
torchvision                   0.9.0a0
```

居然还有conda环境不一致的问题

``` 
root@master:/workspace# conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch -y
Collecting package metadata (current_repodata.json): done
Solving environment: failed with initial frozen solve. Retrying with flexible solve.
Collecting package metadata (repodata.json): done
Solving environment: |
The environment is inconsistent, please check the package plan carefully
The following packages are causing the inconsistency:

  - defaults/linux-64::numpy==1.19.2=py38h6163131_0
  - defaults/linux-64::ipython==7.21.0=py38hb070fc8_0
  - defaults/noarch::pygments==2.8.1=pyhd3eb1b0_0
  - defaults/linux-64::jsonschema==3.0.2=py38_0
  - defaults/linux-64::cython-blis==0.7.4=py38h27cfd23_1
  - defaults/noarch::prompt-toolkit==3.0.8=py_0
  - defaults/noarch::jinja2==2.11.3=pyhd3eb1b0_0
  - defaults/linux-64::scipy==1.6.1=py38hf56f3a7_0
  - defaults/linux-64::conda==4.9.2=py38h06a4308_0
  - defaults/linux-64::thinc==7.4.5=py38h9a67853_0
  - defaults/linux-64::conda-package-handling==1.7.2=py38h03888b9_0
  - defaults/linux-64::numba==0.52.0=py38ha9443f7_0
  - defaults/linux-64::spacy==2.3.5=py38hff7bd54_0
  - defaults/linux-64::conda-build==3.21.4=py38h06a4308_0                                                                                                   done


==> WARNING: A newer version of conda exists. <==
  current version: 4.9.2
  latest version: 25.1.1

Please update conda by running

    $ conda update -n base -c defaults conda



## Package Plan ##

  environment location: /opt/conda

  added / updated specs:
    - pytorch==1.12.1
    - torchaudio==0.12.1
    - torchvision==0.13.1


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    _openmp_mutex-5.1          |            1_gnu          21 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    ca-certificates-2024.12.31 |       h06a4308_0         128 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    certifi-2024.8.30          |   py38h06a4308_0         162 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    charset-normalizer-3.3.2   |     pyhd3eb1b0_0          44 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    cudatoolkit-11.3.1         |       h2bc3f7f_2       549.3 MB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    ffmpeg-4.3                 |       hf484d3e_0         9.9 MB  pytorch
    freetype-2.12.1            |       h4a9f257_0         626 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    future-0.18.3              |   py38h06a4308_0         672 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    gmp-6.3.0                  |       h6a678d5_0         608 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    gnutls-3.6.15              |       he1e5248_0         1.0 MB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    jpeg-9e                    |       h5eee18b_3         262 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    lame-3.100                 |       h7b6447c_0         323 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    lcms2-2.12                 |       h3be6417_0         312 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    libgcc-ng-11.2.0           |       h1234567_1         5.3 MB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    libgomp-11.2.0             |       h1234567_1         474 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    libiconv-1.16              |       h5eee18b_3         759 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    libidn2-2.3.4              |       h5eee18b_0         146 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    libpng-1.6.39              |       h5eee18b_0         304 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    libprotobuf-3.20.3         |       he621ea3_0         2.4 MB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    libstdcxx-ng-11.2.0        |       h1234567_1         4.7 MB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    libtasn1-4.19.0            |       h5eee18b_0          63 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    libtiff-4.2.0              |       h85742a9_0         502 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    libunistring-0.9.10        |       h27cfd23_0         536 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    libwebp-base-1.3.2         |       h5eee18b_1         425 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    nettle-3.7.3               |       hbbd107a_1         809 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    numpy-base-1.19.2          |   py38h75fe3a5_0         4.2 MB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    openh264-2.1.1             |       h4ff587b_0         711 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    openjpeg-2.4.0             |       h9ca470c_2         363 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    openssl-1.1.1w             |       h7f8727e_0         3.7 MB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    pillow-10.4.0              |   py38h5eee18b_0         795 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    pytorch-1.12.1             |cpu_py38h9dbd814_1        49.2 MB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    pytorch-mutex-1.0          |             cuda           3 KB  pytorch
    requests-2.32.3            |   py38h06a4308_0         100 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    setuptools-75.1.0          |   py38h06a4308_0         1.7 MB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    torchaudio-0.12.1          |       py38_cu113         6.2 MB  pytorch
    torchvision-0.13.1         |       py38_cu113        28.7 MB  pytorch
    tqdm-4.66.5                |   py38h2f386ee_0         133 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    typing-extensions-4.11.0   |   py38h06a4308_0           9 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    typing_extensions-4.11.0   |   py38h06a4308_0          59 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    zlib-1.2.13                |       h5eee18b_1         111 KB  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    ------------------------------------------------------------
                                           Total:       675.5 MB

The following NEW packages will be INSTALLED:

  _openmp_mutex      anaconda/pkgs/main/linux-64::_openmp_mutex-5.1-1_gnu
  charset-normalizer anaconda/pkgs/main/noarch::charset-normalizer-3.3.2-pyhd3eb1b0_0
  cudatoolkit        anaconda/pkgs/main/linux-64::cudatoolkit-11.3.1-h2bc3f7f_2
  ffmpeg             pytorch/linux-64::ffmpeg-4.3-hf484d3e_0
  freetype           anaconda/pkgs/main/linux-64::freetype-2.12.1-h4a9f257_0
  future             anaconda/pkgs/main/linux-64::future-0.18.3-py38h06a4308_0
  gmp                anaconda/pkgs/main/linux-64::gmp-6.3.0-h6a678d5_0
  gnutls             anaconda/pkgs/main/linux-64::gnutls-3.6.15-he1e5248_0
  jpeg               anaconda/pkgs/main/linux-64::jpeg-9e-h5eee18b_3
  lame               anaconda/pkgs/main/linux-64::lame-3.100-h7b6447c_0
  lcms2              anaconda/pkgs/main/linux-64::lcms2-2.12-h3be6417_0
  libgomp            anaconda/pkgs/main/linux-64::libgomp-11.2.0-h1234567_1
  libiconv           anaconda/pkgs/main/linux-64::libiconv-1.16-h5eee18b_3
  libidn2            anaconda/pkgs/main/linux-64::libidn2-2.3.4-h5eee18b_0
  libpng             anaconda/pkgs/main/linux-64::libpng-1.6.39-h5eee18b_0
  libprotobuf        anaconda/pkgs/main/linux-64::libprotobuf-3.20.3-he621ea3_0
  libtasn1           anaconda/pkgs/main/linux-64::libtasn1-4.19.0-h5eee18b_0
  libtiff            anaconda/pkgs/main/linux-64::libtiff-4.2.0-h85742a9_0
  libunistring       anaconda/pkgs/main/linux-64::libunistring-0.9.10-h27cfd23_0
  libwebp-base       anaconda/pkgs/main/linux-64::libwebp-base-1.3.2-h5eee18b_1
  nettle             anaconda/pkgs/main/linux-64::nettle-3.7.3-hbbd107a_1
  numpy-base         anaconda/pkgs/main/linux-64::numpy-base-1.19.2-py38h75fe3a5_0
  openh264           anaconda/pkgs/main/linux-64::openh264-2.1.1-h4ff587b_0
  openjpeg           anaconda/pkgs/main/linux-64::openjpeg-2.4.0-h9ca470c_2
  pillow             anaconda/pkgs/main/linux-64::pillow-10.4.0-py38h5eee18b_0
  pytorch            anaconda/pkgs/main/linux-64::pytorch-1.12.1-cpu_py38h9dbd814_1
  pytorch-mutex      pytorch/noarch::pytorch-mutex-1.0-cuda
  requests           anaconda/pkgs/main/linux-64::requests-2.32.3-py38h06a4308_0
  setuptools         anaconda/pkgs/main/linux-64::setuptools-75.1.0-py38h06a4308_0
  torchaudio         pytorch/linux-64::torchaudio-0.12.1-py38_cu113
  torchvision        pytorch/linux-64::torchvision-0.13.1-py38_cu113
  tqdm               anaconda/pkgs/main/linux-64::tqdm-4.66.5-py38h2f386ee_0
  typing-extensions  anaconda/pkgs/main/linux-64::typing-extensions-4.11.0-py38h06a4308_0
  typing_extensions  anaconda/pkgs/main/linux-64::typing_extensions-4.11.0-py38h06a4308_0

The following packages will be UPDATED:

  ca-certificates    pkgs/main::ca-certificates-2021.1.19-~ --> anaconda/pkgs/main::ca-certificates-2024.12.31-h06a4308_0
  certifi            pkgs/main::certifi-2020.12.5-py38h06a~ --> anaconda/pkgs/main::certifi-2024.8.30-py38h06a4308_0
  libgcc-ng           pkgs/main::libgcc-ng-9.1.0-hdf63c60_0 --> anaconda/pkgs/main::libgcc-ng-11.2.0-h1234567_1
  libstdcxx-ng       pkgs/main::libstdcxx-ng-9.1.0-hdf63c6~ --> anaconda/pkgs/main::libstdcxx-ng-11.2.0-h1234567_1
  openssl              pkgs/main::openssl-1.1.1j-h27cfd23_0 --> anaconda/pkgs/main::openssl-1.1.1w-h7f8727e_0
  zlib                    pkgs/main::zlib-1.2.11-h7b6447c_3 --> anaconda/pkgs/main::zlib-1.2.13-h5eee18b_1



Downloading and Extracting Packages
libunistring-0.9.10  | 536 KB    | ################################################################################################################### | 100%
lcms2-2.12           | 312 KB    | ################################################################################################################### | 100%
freetype-2.12.1      | 626 KB    | ################################################################################################################### | 100%
setuptools-75.1.0    | 1.7 MB    | ################################################################################################################### | 100%
charset-normalizer-3 | 44 KB     | ################################################################################################################### | 100%
ffmpeg-4.3           | 9.9 MB    | ################################################################################################################### | 100%
libtiff-4.2.0        | 502 KB    | ################################################################################################################### | 100%
pytorch-1.12.1       | 49.2 MB   | ################################################################################################################### | 100%
jpeg-9e              | 262 KB    | ################################################################################################################### | 100%
torchaudio-0.12.1    | 6.2 MB    | ################################################################################################################### | 100%
openjpeg-2.4.0       | 363 KB    | ################################################################################################################### | 100%
numpy-base-1.19.2    | 4.2 MB    | ################################################################################################################### | 100%
openssl-1.1.1w       | 3.7 MB    | ################################################################################################################### | 100%
libpng-1.6.39        | 304 KB    | ################################################################################################################### | 100%
libgomp-11.2.0       | 474 KB    | ################################################################################################################### | 100%
zlib-1.2.13          | 111 KB    | ################################################################################################################### | 100%
lame-3.100           | 323 KB    | ################################################################################################################### | 100%
requests-2.32.3      | 100 KB    | ################################################################################################################### | 100%
cudatoolkit-11.3.1   | 549.3 MB  | ################################################################################################################### | 100%
certifi-2024.8.30    | 162 KB    | ################################################################################################################### | 100%
typing-extensions-4. | 9 KB      | ################################################################################################################### | 100%
ca-certificates-2024 | 128 KB    | ################################################################################################################### | 100%
torchvision-0.13.1   | 28.7 MB   | ################################################################################################################### | 100%
libgcc-ng-11.2.0     | 5.3 MB    | ################################################################################################################### | 100%
nettle-3.7.3         | 809 KB    | ################################################################################################################### | 100%
libstdcxx-ng-11.2.0  | 4.7 MB    | ################################################################################################################### | 100%
typing_extensions-4. | 59 KB     | ################################################################################################################### | 100%
tqdm-4.66.5          | 133 KB    | ################################################################################################################### | 100%
openh264-2.1.1       | 711 KB    | ################################################################################################################### | 100%
pytorch-mutex-1.0    | 3 KB      | ################################################################################################################### | 100%
_openmp_mutex-5.1    | 21 KB     | ################################################################################################################### | 100%
libidn2-2.3.4        | 146 KB    | ################################################################################################################### | 100%
libtasn1-4.19.0      | 63 KB     | ################################################################################################################### | 100%
gnutls-3.6.15        | 1.0 MB    | ################################################################################################################### | 100%
future-0.18.3        | 672 KB    | ################################################################################################################### | 100%
libwebp-base-1.3.2   | 425 KB    | ################################################################################################################### | 100%
pillow-10.4.0        | 795 KB    | ################################################################################################################### | 100%
libprotobuf-3.20.3   | 2.4 MB    | ################################################################################################################### | 100%
libiconv-1.16        | 759 KB    | ################################################################################################################### | 100%
gmp-6.3.0            | 608 KB    | ################################################################################################################### | 100%
Preparing transaction: done
Verifying transaction: failed

RemoveError: 'requests' is a dependency of conda and cannot be removed from
conda's operating environment.
RemoveError: 'setuptools' is a dependency of conda and cannot be removed from
conda's operating environment.
```

在卡死的不知道为什么只有3个进程，正常的时候是26个。


# 实验方案
三个东西作为输入，分为直接重新重建，自回归，多输出

以及半思维链，半自回归，半强化学习，最小编辑距离作为reward

关于loss下降太慢的问题，可能还是有点问题，要不考虑，将所有能够促使loss下降的梯度方向累积下来，做滑动平均，舍弃那些垃圾的梯度方向。也就是不只是单纯的给整体的梯度来算东西。
而是除了历史梯度以外，还要考虑以往的loss反馈。
但是历史梯度的存储非常费显存，估计是要做多进程的硬盘或内存转显存的操作