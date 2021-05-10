# sanskrit-ocr

*Note: This branch contains code for IndicOCR-v2. For IndicOCR-v1, kindly visit the `master` branch.*


----------------------------------------------------------------------------

This repository contains code for various OCR models for classical Sanskrit Document Images. For a quick understanding of how to get the **IndicOCR** and **CNN-RNN** up and running, kindly continue to read this Readme. For more detailed instructions, visit our  [Wiki](https://github.com/ihdia/sanskrit-ocr/wiki) page.

The IndicOCR model and CNN-RNN models are best run on a GPU.

Please cite [our paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w34/Dwivedi_An_OCR_for_Classical_Indic_Documents_Containing_Arbitrarily_Long_Words_CVPRW_2020_paper.pdf) if you end up using it for your own research.


```
@InProceedings{Dwivedi_2020_CVPR_Workshops,
author = {Dwivedi, Agam and Saluja, Rohit and Kiran Sarvadevabhatla, Ravi},
title = {An OCR for Classical Indic Documents Containing Arbitrarily Long Words},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2020}
}
```
-----------------------------------------------------------------------------------------------------

# Results:

The following table shows the comparitive results for the IndicOCR-v2 model with other state of the art models.

|Row |**Dataset**|**Model**|Training Config| CER (%)| WER (%)|
|----|-------------|--------|----------|-----|----|
1 | new | IndicOCR-v2 | C3:mix training + real finetune |3.86 |13.86|
2 | new | IndicOCR-v2 | C1:mix training  |4.77|16.84|
3 | new | CNN-RNN | C3:mix training + real finetune |3.77| 14.38|
4 | new | CNN-RNN | C1:mix training | 3.67| 13.86|
5 | new | Google-OCR | -- | 6.95| 34.64| 
6 | new | Ind.senz | -- | 20.55| 57.92| 
7 | new | Tesseract (Devanagiri)| --  |13.23 |52.75| 
8 | new | Tesseract (Sanskrit)| --  |21.06 |62.34| 

## IndicOCR-v2:

### Details:

The code is written in **tensorflow** framework.

### Pre-Trained Models:

To download our best models, kindly visit this page.

### Setup:

In the model/attention-lstm directory, run the following commands:
```
create conda create -n indicOCR python=3.6.10
conda activate indicOCR
conda install pip
pip install -r requirements.txt
```

### Installation:
To install the `aocr` (attention-ocr) library, from the model/attention-lstm directory, run:

```
python setup.py install
```

#### tfrecords creation:

Make sure to have/create a `.txt` file with every line of the file in the following format:

`path/to/image<space>annotation`

**ex:** `/user/sanskrit-ocr/datasets/train/1.jpg I am the annotated text`

```
aocr dataset /path/to/txt/file/ /path/to/data.tfrecords
```

### Train:

To train the `data.tfrecords` file created as described above, run the following command.

```
CUDA_VISIBLE_DEVICES=0 aocr train /path/to/tfrecords/file --batch-size <batch-size> --max-width <max-width> --max-height <max-height> --max-prediction <max-predicted-label-length> --num-epoch <num-epoch>
```

### Validate:

To validate many checkpoints, run 

```
python ./model/evaluate/attention_predictions.py <initial_ckpt_no> <final_ckpt_step> <steps_per_checkpoint>
```

This will create a `val_preds.txt` file in the model/attention-lstm/logs folder.

### Test

To test a single checkpoint, run the following command:

```
CUDA_VISIBLE_DEVICES=0 aocr test /path/to/test.tfrecords --batch-size <batch-size> --max-width <max-width> --max-height <max-height> --max-prediction <max-predicted-label-length> --model-dir ./modelss
```

*Note: If you want to test multiple checkpoints which are evenly spaced (numbering wise), use the method described in the [validation](#Validate) section.*

### Computing Error Rates:

To compute the CER and WER of the predictions, run the following command:

```
python ./model/evaluate/get_errorrates.py <predicted_file_name>

```

**ex:** `python model/evaluate/get_errorrates.py val_preds.txt`

The results of error rates will be written to a file `output.json` in the `visualize` directory.

## CNN-RNN:

### Details:

The code is written in **tensorflow** framework.

### Pre-Trained Models:

To download the best CNN-RNN model, kindly visit this page.

### Setup:

In the model/CNN-RNN directory, run the following commands:
```
create conda create -n crnn python=3.6.10
conda activate crnn
conda install pip
pip install -r requirements.txt
```

#### tfrecords creation:

Make sure to have/create a `.txt` file with every line of the file in the following format:

`path/to/image<space>annotation`

**ex:** `/user/sanskrit-ocr/datasets/train/1.jpg I am the annotated text`

```
python model/CRNN/create_tfrecords.py /path/to/.txt/file ./model/CRNN/data/tfReal/data.tfrecords
```

### Train:

To train the `data.tfrecords` file created as described above, run the following command.

```
python model/CRNN/train.py <training tfrecords filename> <train_epochs> <path_to_previous_saved_model> <steps-per_checkpoint>
```
**ex:** `python ./model/CRNN/train.py train_feature.tfrecords 20 model/CRNN/model/shadownet/shadownet_-40 200`

*Note: If you are training from scratch just set the `<path_to_previous_saved_model>` arguement to 0.*

**ex:** `python model/CRNN/train.py data.tfrecords 100 0 <steps-per_checkpoint>`

### Validate:

To validate many checkpoints, run 

```
python ./model/evaluate/crnn_predictions.py <initial_step> <final_step> <steps_per_checkpoint>
```

This will create a `val_preds.txt` file in the model/attention-lstm/logs folder.

### Test

To test a single checkpoint, run the following command:

```
CUDA_VISIBLE_DEVICES=0 aocr test /path/to/test.tfrecords --batch-size <batch-size> --max-width <max-width> --max-height <max-height> --max-prediction <max-predicted-label-length> --model-dir ./modelss
```

*Note: If you want to test multiple checkpoints which are evenly spaced (numbering wise), use the method described in the [validation](#Validate) section.*

### Computing Error Rates:

To compute the CER and WER of the predictions, run the following command:

```
python ./model/evaluate/get_errorrates.py <predicted_file_name>

```

**ex:** `python model/evaluate/get_errorrates.py val_preds.txt`

The results of error rates will be written to a file `output.json` in the `visualize` directory.