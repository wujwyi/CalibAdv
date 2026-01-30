
DATA_NAME=hotpotqa

DATASET_PATH=data/${DATA_NAME}
save_dir=
# save_dir=${DATASET_PATH}

SPLIT=dev
TOPK=10

INDEX_PATH=
CORPUS_PATH=
retriever_model=
save_path_file=${save_dir}/${DATA_NAME}_${SPLIT}_top${TOPK}.json

# INDEX_PATH=/home/peterjin/rm_retrieval_corpus/index/wiki-21
# CORPUS_PATH=/home/peterjin/rm_retrieval_corpus/corpora/wiki/enwiki-dec2021/text-list-100-sec.jsonl
# SAVE_NAME=e5_${TOPK}_wiki21.json

CUDA_VISIBLE_DEVICES=0,1,2,3 python search_r1/search/retrieval.py \
                    --retrieval_method e5 \
                    --retrieval_topk $TOPK \
                    --index_path $INDEX_PATH \
                    --corpus_path $CORPUS_PATH \
                    --dataset_path $DATASET_PATH \
                    --data_split $SPLIT \
                    --retrieval_model_path $retriever_model \
                    --retrieval_pooling_method "mean" \
                    --retrieval_batch_size 512 \
                    --save_path_file $save_path_file \
