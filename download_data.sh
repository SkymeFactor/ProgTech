mkdir data
cd data

wget -c http://ufldl.stanford.edu/housenumbers/train_32x32.mat http://ufldl.stanford.edu/housenumbers/test_32x32.mat

if [ -e "./test_32x32.mat" ] || [[ -e "./train_32x32.mat" ]]; then

    python ../Extractor/Extractor.py

    if [ $? -ne 0 ]; then
        echo -e "\nContinuing with python 3\n"
        python3 ../Extractor/Extractor.py
        if [ $? -ne 0 ]; then
            echo -e "\nExtraction failed\n"
        fi
    fi
else
    echo -e "\nSome of the archives are missing, aborted\n"
fi