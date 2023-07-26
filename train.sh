python train.py --fold 0 \
--pretrained_path ./PhoBERT_base_transformers/model.bin \
--config_path ./PhoBERT_base_transformers/config.json \
--dict_path ./PhoBERT_base_transformers/dict.txt \
--data_path ./data/data_segment.csv \
--ckpt_path ./ckpt \
--bpe_codes /content/sentiment_analysis/PhoBERT_base_transformers/bpe.codes