#!/bin/bash

conda init bash
#source ~/.bashrc
conda activate topoml
cd /home/sam/Documents/PhD/Research/getognn

# python ./TrainMSCGNN.py training_selection_method dataset depth growing_windows_line

## input
#x_1 = sys.argv[1]
#x_2 = sys.argv[2]
#y_1 = sys.argv[3]
#y_2 = sys.argv[4]
# dataset id
run=$(cat ./run_count.txt).log
increment=1
run=$((run+increment))
time=`date +"%k-%M"`
python ./supervised_train.py |& tee -a ./log-dir/run_"$run"_"$time".log
#for i in {0..0}
#do
#    for j in {0..35}
#    do
#    #window=$((69 * i))
#    #start=70
#    #step=$((start + window))
#    #python TrainMSCGNN.py 200 400 300 500 |& tee -a ./log-dir/$(cat ../run_count.txt).log
#    #                    read_param sample
#
#	python ./TrainMSCGNN.py 1 "$i" 3 "$j"|& tee -a ./log-dir/"$run".log
#    done
#done
