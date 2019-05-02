mkdir data
cd data

# The German-English translation task of the IWSLT evaluation campaign
mkdir iwslt
cd iwslt
wget http://phontron.com/data/iwslt-en-de-preprocessed.tar.gz
tar -xf iwslt-en-de-preprocessed.tar.gz
cd ..

# fasttext https://fasttext.cc pre-trained vector embedding
mkdir fasttext
cd fasttext
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
unzip wiki-news-300d-1M.vec.zip
cd ..
