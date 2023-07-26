# Sentiment Analysis

## Install Java for vncorenlp library word segment
```
sudo apt update -y && sudo apt upgrade -y
sudo apt install default-jdk -y
sudo apt install default-jre -y
```
## Install & Download vncorenlp model
```
pip install py_vncorenlp
py_vncorenlp.download_model(save_dir='/absolute/path/to/vncorenlp')
```

## Requirements
```
cd sentiment_analysis
pip install -r requirements.txt
```
## Dowloading PhoBERT
```
wget https://public.vinai.io/PhoBERT_base_transformers.tar.gz
tar -xzvf PhoBERT_base_transformers.tar.gz
```

## Train
```
python train.py --fold <fold_id> \
--dict_path <phobert dict path>\
--config_path <path to config> \
--data_path <data path> \
--ckpt_path <ckpt path> \
--bpe_codes <absolute path to bpe>
```
