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
* [ ] add helpful function for tensorboard-style summary writer
* [ ] analyze the reason of low GPU utilizing
* [ ] analyze the situation and the modifying necessity of err `list index out of range`
* [ ] study the argument `n_head_kv` of Llama3, why it will influence the success of model building
* [ ] add support of AMP
* [ ] rethink the variable `self.change_file_event` and try using it to resolve `list index out of range`
* [ ] resolve the problem of process exiting failed
* [ ] modify the file_segments_dataloader with a start switch, not a automatical start after initializing
* [ ] add cpu support for debugging purpose