# Anomaly-Detection

project structure

Anomaly-Detection/
    config/
        schema
    Data/
    archive.zip // da awel dataset fel proposal ghaleban msh hanehtag gherha
    src/
    main.ipynb

# create virtual environment 
# kolo on WSL 
python3 -m venv SparkEnv

source SparkEnv/bin/activate

pip install install pyspark 

# lazem teb2a menazel java w hatet el directory mazboot fel bashrc

unzip archive.zip 

# run ipynb momken men anaconda aw vs code lazem tehotto kernel SparkEnv lama yes2alak 3aleha



--datasets
two that noor used:
https://www.kaggle.com/code/adepvenugopal/logs-dataset/input?select=access_log.txt
https://www.kaggle.com/datasets/eliasdabbas/web-server-access-logs
linux suggestion--
https://www.kaggle.com/code/adepvenugopal/logs-dataset/input?select=Linux.log 

--webserver preprocessing
beyhot regex 3la hasab structure kol dataset betala3 logs ezay
fi wahda mell datasets fiha bots fa claude edany elkeywords ely ne3raf biha en howa bot 
then bey clean data into columns 
fi dataset menhom mafihash hetet el users w elbots di fa elcolumns di hatebaa null/0
then bey validate against schema and merges both cleaned datasets into one dataset w ba print statistics kda
