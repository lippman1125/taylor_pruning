## PyTorch implementation of  [\[1611.06440 Pruning Convolutional Neural Networks for Resource Efficient Inference\]](https://arxiv.org/abs/1611.06440) ##

This demonstrates pruning a VGG16 based classifier that classifies a small dog/cat dataset.


This was able to reduce the CPU runtime by x3 and the model size by x4.

For more details you can read the [blog post](https://jacobgil.github.io/deeplearning/pruning-deep-learning).

At each pruning step 512 filters are removed from the network.


Usage
-----

This repository uses the PyTorch ImageFolder loader, so it assumes that the images are in a different directory for each category.

Train

......... dogs

......... cats


Test


......... dogs

......... cats


The images were taken from [here](https://www.kaggle.com/c/dogs-vs-cats) but you should try training this on your own data and see if it works!  
1. We need a baseline model, if we already have one, we do not need training.  
Training:  
`python finetune.py --train`  
2. Pruning during finetune.  
Pruning:  
`python finetune.py --prune`
3. Pruning step  
  step1. setup finetune dataset, sub from traindata  
  step2. set num_filters_to_prune_per_iteration, we can prune multi-filters one iteration (one epoch)  
  step3. collecting the importance of the channel (average grad*activation), one channel is the outputs of one prev layer's filter  
  step4. normalize the importance  
  step5. remove least important filter  
  step6. repeat 3-5   

TBD
---

 - Change the pruning to be done in one pass. Currently each of the 512 filters are pruned sequentually. 
	`
	for layer_index, filter_index in prune_targets:
			model = prune_vgg16_conv_layer(model, layer_index, filter_index)
		`


 	This is inefficient since allocating new layers, especially fully connected layers with lots of parameters, is slow.
	
	In principle this can be done in a single pass.



 - Change prune_vgg16_conv_layer to support additional architectures.
 	The most immediate one would be VGG with batch norm.

