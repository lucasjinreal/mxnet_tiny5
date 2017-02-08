### Mxnet Train Your Own Data For Classify Task

### Data
For before training, mxnet recommended using rec for input imagedata, in early mxnet version we have `ImageRecordIter` and `DataIter` to load your images, in recent update of mxnet, they are merged into `ImageIter` ,you can load your image in `rec` format as well as original images. But in this tutorial we are going using `ImageRecordIter` .

* generate image list

`im2rec_gen_list.sh` can generate `train_list.txt` and `val_list.txt`, each file contains content like this:
```
image_index class_index class_folder/image_name.jpg
```

before you run this script you must have `map_class_index.txt` file in this format:
```
horse 0
flower 1
elephant 2
dinosaur 3
bus 4
```
**[UPDATE] 2017-2-1**
To generate mxnet im2rec needed list file now we have `im2rec_gen_list.py` , you can use it like this:
```
python3 im2rec_gen_list.py -train=/media/work/jfg/MxnetSpace/mxnet_classification/Tiny5/train -val=/media/work/jfg/MxnetSpace/mxnet_classification/Tiny5/test -shuffle=True
```
this will generate train_list.txt and val_list.txt, set shuffle True will random shuffle train images.

* generate rec file

the rest is easy, using `im2rec` binary program you just need type:
```
./im2rec train_list_txt /media/work/jfg/images tiny5_train.rec resize=100x150
```
**parma1** your train_list file
**param2** your full image root path
**param3** your save file
and the last your resize shape, in **height x width** format.

> One thing have to be aware, `resize` also can be set to single value like 100, this means image will shrink long dim to 100, for example original image is width*height=334*225, if resize=100, it will make width to 100 and height will be (225/334) * 100, that is to say all dims will limited to under 100.

### Build Mxnet Symbols
<<<<<<< HEAD
=======

----------------------------------------------------华丽里的分割线------------------------------------------------------------
### 中文说明
这个系列是我mxnet上手的记录，也是大家用一个框架必须经历的过程，比如我们实现一个LeNet，然后用自己的数据来训练连它，最后用它来分类这样才能学一直用，而本系列教程便是实现这个目的而来的。
上面的英文部分说明了如何生成list文件，但是在我这个repo中mxnet官方的二进制文件工具`im2rec` 这个文件太大我就没有放上来了，用到的tiny5这个数据集也是我自己做的，你有可以在我的另一个repo：caffe_tiny5中找到数据集的链接，下载即可。

### mxnet已实现断点续训
mxnet灵活的地方就在于框架给你但是你要自己去训练他，甚至checkpoint这样的东西你都要自己去实现，通常你只知道用mxnet去训练然后保存模型，最后手动从一个地方开始重新训练，在本教程中你不需要，都已经实现好了，你在任何地方ctral + C，再次运行时会立马从原来停下来的地方重新训练，可以节省我们宝贵的时间。

### mxnet下一步预告
这只是一个简单的实例，相信对于大家入门来说还是很有用的，接下来我会实现一个基于mxnet的SSD的教程，并用它来实现一些东西，甚至搬到移动端来。
>>>>>>> 35c6f35fc71bd453592af0cd1d936257506e1abe
