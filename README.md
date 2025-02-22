# dlx

An open source project to customize your own Large Language Model, designed for modifying and learning easily to against 
current terrible ambiance which the code is complex and opaque. The most teams always only release the inference code, 
without pretraining code. Such as, Openai and Meta-Llama.

If possible, the models of Computer Vision and Multi-modal will be added, too.

## highlights

* powerful and diverse dataloader implementation with multi-processes and multi-threads, such as 
`FileSegmentsDataloader` to process multi-files under a certain folder. 

## test projects

The projects under the module `dlx.test`.

You can also have a quick experience by run 
`torchrun --nproc_per_node 1 experiments/llama3/pretrain.py`

## todo 
* [ ] add another method for file_segments_dataloader to generate sample list.
* [x] add helpful function for tensorboard-style summary writer
* [x] analyze the reason of low GPU utilizing
* [x] analyze the situation and the modifying necessity of err `list index out of range`
* [x] study the argument `n_head_kv` of Llama3, why it will influence the success of model building
* [x] add support of AMP
* [x] rethink the variable `self.change_file_event` and try using it to resolve `list index out of range`
* [ ] resolve the problem of process exiting failed
* [ ] modify the file_segments_dataloader with a start switch, not an automatical start after initializing
* [ ] add cpu support for debugging purpose
* [ ] add token calculation for statistic
* [ ] 调研其他小的LLM的tokens数量
* [ ] 搞定deepseek和ddp的接口整合


## Documents

### utils.data.dataloader

#### DataLoader

If the method `__len__` is not implemented, the step will be set to a number.

现在的问题是总会丢掉最后一个样本，且不好修改这部分的逻辑。