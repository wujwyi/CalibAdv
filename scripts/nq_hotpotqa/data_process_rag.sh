WORK_DIR=./
LOCAL_DIR=$WORK_DIR/data/nq_hotpotqa_rag_top5

DATA=nq,hotpotqa
retrieval_caches_dir=
python $WORK_DIR/scripts/data_process/qa_rag_train_merge.py --topk 5 --local_dir $LOCAL_DIR --data_sources $DATA --retrieval_caches_dir $retrieval_caches_dir

# DATA=nq,hotpotqa
# retrieval_caches_dir=
python $WORK_DIR/scripts/data_process/qa_rag_test_merge.py --topk 5 --local_dir $LOCAL_DIR --data_sources $DATA  --retrieval_caches_dir $retrieval_caches_dir
