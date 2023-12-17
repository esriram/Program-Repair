#!/usr/bin/env bash

function prompt() {
    echo "Syntax: bash concode.sh <GPU_ID> <EXP_NAME> <CKPT_NUM>";
    exit;
}

while getopts ":h" option; do
    case $option in
        h) # display help
          prompt;
    esac
done

if [[ $# -ne 3 ]]; then
    prompt;
fi

GPU=$1;
EXP_NAME=$2;
CKPT_NUM=$3;
CODE_HOME_DIR='.';
DATA_DIR="${CODE_HOME_DIR}/resources/finetuning/data";

function download() {
    mkdir -p ${DATA_DIR}/concode;
    cdir='pwd';
    cd $DATA_DIR/concode;
    gdown 1kC6fe7JgOmEHhVFaXjzOmKeatTJy1I1W;
    cd ${cdir};
}

function finetune() {
    OUTPUT_DIR="${CODE_HOME_DIR}/resources/finetuning/models/concode";
    mkdir -p ${OUTPUT_DIR};
    LOG="${OUTPUT_DIR}/finetuning.log";
    SUMMARY_DIR="${OUTPUT_DIR}/summary";
    mkdir -p ${SUMMARY_DIR};
    CACHE_DIR="${OUTPUT_DIR}/cache";
    mkdir -p ${CACHE_DIR};
    RES_DIR="${OUTPUT_DIR}/results";
    mkdir -p $RES_DIR;

    PRETEAINED_MODEL_BASE="${CODE_HOME_DIR}/resources/pretraining/models";
    PRETRAINING_EXP_NAME="${EXP_NAME}"
    PRETRAINED_MODEL_NAME="checkpoint-${CKPT_NUM}";
    PRETRAINED_MODEL_PATH="${PRETEAINED_MODEL_BASE}/${PRETRAINING_EXP_NAME}/${PRETRAINED_MODEL_NAME}";
    if [ $CKPT_NUM == "0" ]; then
        PRETRAINED_MODEL_PATH="Salesforce/codet5-base";
    fi

    export PYTHONIOENCODING=utf-8;
    export PYTHONPATH=$PYTHONPATH:$CODE_HOME_DIR;
    SCRIPT_PATH="${CODE_HOME_DIR}/src/finetuning/generation.py";

    export CUDA_VISIBLE_DEVICES=${GPU};

    BATCH_SIZE=16;
    GRADIENT_ACCUM_STEP=1;
    NUM_EPOCHS=30;
    PATIENCE=15;
    LEARNING_RATE=0.0001;
    SRC_LEN=256;
    TGT_LEN=256;

    python3 $SCRIPT_PATH \
            --do_train --do_eval --do_eval_bleu --do_test \
            --task concode --sub_task none \
            --model_type codet5 \
            --data_num -1  \
            --warmup_steps 1000 \
            --learning_rate ${LEARNING_RATE} \
            --num_train_epochs ${NUM_EPOCHS} \
            --patience ${PATIENCE} \
            --tokenizer_name ${PRETRAINED_MODEL_PATH}  \
            --model_name_or_path ${PRETRAINED_MODEL_PATH} \
            --data_dir ${DATA_DIR}  \
            --cache_path ${CACHE_DIR}  \
            --output_dir ${OUTPUT_DIR}  \
            --summary_dir ${SUMMARY_DIR} \
            --save_last_checkpoints \
            --always_save_model \
            --res_dir ${RES_DIR} \
            --res_fn ${RES_DIR}/results.txt \
            --train_batch_size ${BATCH_SIZE} \
            --gradient_accumulation_steps ${GRADIENT_ACCUM_STEP} \
            --eval_batch_size ${BATCH_SIZE} \
            --max_source_length ${SRC_LEN} \
            --max_target_length ${TGT_LEN} 2>&1 | tee ${LOG}
}

# download;
finetune;
