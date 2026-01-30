WORK_DIR=./
LOCAL_DIR=$WORK_DIR/data/hotpotqa_2wiki_searchr1

# # process multiple dataset search format train file
DATA=hotpotqa,2wikimultihopqa
python3 $WORK_DIR/scripts/data_process/qa_search_train_merge.py --local_dir $LOCAL_DIR --data_sources $DATA

# ## process multiple dataset search format test file
# DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
# python3 $WORK_DIR/scripts/data_process/qa_search_test_merge.py --local_dir $LOCAL_DIR --data_sources $DATA

# DATA=hotpotqa,2wikimultihopqa
# python3 $WORK_DIR/scripts/data_process/qa_search_test_merge.py --local_dir $LOCAL_DIR --data_sources $DATA
