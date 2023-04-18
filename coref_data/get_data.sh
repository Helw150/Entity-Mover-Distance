# GET ECB+ DATA IN CORRECT FORMAT

wget https://github.com/cltl/ecbPlus/raw/master/ECB%2B_LREC2014/ECB%2B.zip
unzip ECB+.zip
rm -rf __MACOSX/
wget https://raw.githubusercontent.com/cltl/ecbPlus/master/ECB%2B_LREC2014/ECBplus_coreference_sentences.csv
rm -rf ECB+/.DS_Store
python make_dataset.py --ecb_path ECB+ --output_dir ecb/interim --data_setup 2 --selected_sentences_file ECBplus_coreference_sentences.csv
rm -rf ECB+
rm ECB+.zip
rm -rf ECBplus_coreference_sentences.csv
python build_features.py --config_path ecb_config.json --output_path ecb/final
rm -rf ecb/interim

# GET CD2CR DATA

mkdir cd2cr
mkdir cd2cr/interim
wget https://raw.githubusercontent.com/ravenscroftj/cdcrtool/master/CDCR_Corpus/train.conll
wget https://raw.githubusercontent.com/ravenscroftj/cdcrtool/master/CDCR_Corpus/dev.conll
wget https://raw.githubusercontent.com/ravenscroftj/cdcrtool/master/CDCR_Corpus/test.conll
mv *.conll cd2cr/interim/
python build_features.py --config_path cd2cr_config.json --output_path cd2cr/final
rm -rf cd2cr/interim
