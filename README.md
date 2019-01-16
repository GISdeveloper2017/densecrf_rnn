# Complex relations in CRF-RNN for Semantic Image Segmentation 

Based on Keras/Tensorflow implementation here: https://github.com/sadeepj/crfasrnn_keras, implemented for the following paper:
```
@inproceedings{crfasrnn_ICCV2015,
    author = {Shuai Zheng and Sadeep Jayasumana and Bernardino Romera-Paredes and Vibhav Vineet and
    Zhizhong Su and Dalong Du and Chang Huang and Philip H. S. Torr},
    title  = {Conditional Random Fields as Recurrent Neural Networks},
    booktitle = {International Conference on Computer Vision (ICCV)},
    year   = {2015}
}
```
Their live demo:</b> &nbsp;&nbsp;&nbsp;&nbsp; [http://crfasrnn.torr.vision](http://crfasrnn.torr.vision) <br/>
Their Caffe version:</b> [http://github.com/torrvision/crfasrnn](http://github.com/torrvision/crfasrnn)<br/>
Paper link:</b> [http://www.robots.ox.ac.uk/~szheng/papers/CRFasRNN.pdf](http://www.robots.ox.ac.uk/~szheng/papers/CRFasRNN.pdf)<br/>

# Usage

python train.py -m <model_name> -ds <dataset_name> -is <input_size> -e <num_epochs> -bs <batch_size> -vb 1 -g <gpu_number>

example

python train.py -m fcn_RESNET50_8s -ds voc2012 -is 224 -e 1 -bs 32 -vb 1 -g 0