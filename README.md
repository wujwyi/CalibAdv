Negative Advantage Is a Double-Edged Sword: Calibrating Advantage in GRPO for Deep Search

![soft_penalty_query_cnt_curve](./soft_penalty_query_cnt_curve.png)
*Figure 1. Number of mis-penalized queries, showing that CalibAdv is most effective in the early stage of training.*

![soft_penalty_query_ratio_curve](./soft_penalty_query_ratio_curve.png)
*Figure 2. The number and ratio of mis-penalized queries among all queries, showing that CalibAdv is most effective in the early stage of training.*

![alt text](all_ans_score_below_0.7_question_ratio_curve.png)
*Figure 3. Fraction of groups in which all trajectories are incorrect, resulting in empty silver sets.*

![alt text](silver_doc_set_boxplot_every10steps.png)
*Figure 4. Distribution of silver set sizes.*

![alt text](cs_score_ratios_curve.png)
*Figure 5. Distribution of query correctness score values, showing that the correctness score does not degenerate into a binary decision.*

![alt text](qwen2.5-3B_f1_lambda.png)
*Figure 6. Impact of the rebalance scaling coefficient λ on Qwen2.5-3B performance dynamics.*

![alt text](qwen2.5-3B_entropy_lambda.png)
*Figure 7. Impact of the rebalance scaling coefficient λ on Qwen2.5-3B entropy dynamics.*

![alt text](llama-3B_f1_lambda.png)
*Figure 8. Impact of the rebalance scaling coefficient λ on Llama-3B-Instruct performance dynamics.*

![alt text](llama-3B_entropy_lambda.png)
*Figure 9. Impact of the rebalance scaling coefficient λ on Llama-3B-Instruct entropy dynamics.*

![alt text](analysis_query_correctness_score_all_experiments.png)
*Figure 10. Query correctness scores across all experiments, showing that Soft Penalty substantially improves search quality.*
