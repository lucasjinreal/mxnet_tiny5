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
