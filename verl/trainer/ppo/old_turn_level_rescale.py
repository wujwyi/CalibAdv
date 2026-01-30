                    # query_x_indices = [[] for _ in range(4)]
                    # query_y_indices = [[] for _ in range(4)]
                    # ans_x_indices = []
                    # ans_y_indices = []
                    # max_query_turn = 0
                    # for idx in range(len(eos_mask)):
                    #     diff = torch.diff(eos_mask[idx], prepend=torch.tensor([0]), append=torch.tensor([0]))
                    #     starts = torch.where(diff == 1)[0]
                    #     ends = torch.where(diff == -1)[0]

                    #     for query_idx in range(len(starts)-1):
                    #         x = torch.full((ends[query_idx] - starts[query_idx],), idx)
                    #         y = torch.arange(starts[query_idx], ends[query_idx])
                    #         query_x_indices[query_idx].append(x)
                    #         query_y_indices[query_idx].append(y)
                    #     max_query_turn = max(max_query_turn, len(starts)-1)

                    #     ans_x = torch.full((ends[-1] - starts[-1],), idx)
                    #     ans_y = torch.arange(starts[-1], ends[-1])
                    #     ans_x_indices.append(ans_x)
                    #     ans_y_indices.append(ans_y)

                    # turn_indices = []
                    # for i in range(max_query_turn):
                    #     query_x_indices[i] = torch.cat(query_x_indices[i])
                    #     query_y_indices[i] = torch.cat(query_y_indices[i])
                    #     turn_indices.append(tuple([query_x_indices[i], query_y_indices[i]]))
                    # ans_x_indices = torch.cat(ans_x_indices)
                    # ans_y_indices = torch.cat(ans_y_indices)
                    # ans_indices = tuple([ans_x_indices, ans_y_indices])
                    # turn_indices.append(ans_indices)

                    # for this_turn_indices in turn_indices:
                    #     this_turn_advantage = scores[this_turn_indices]
                    #     this_turn_positive_mask = this_turn_advantage > 0
                    #     this_turn_negative_mask = this_turn_advantage < 0
                    #     this_turn_postive_indices = (this_turn_indices[0][this_turn_positive_mask], this_turn_indices[1][this_turn_positive_mask])
                    #     this_turn_negative_indices = (this_turn_indices[0][this_turn_negative_mask], this_turn_indices[1][this_turn_negative_mask])
                    #     this_turn_positive_sum = scores[this_turn_postive_indices].sum()
                    #     this_turn_negative_sum = scores[this_turn_negative_indices].sum()
                    #     this_turn_ratio = this_turn_positive_sum / (-this_turn_negative_sum) if this_turn_negative_sum != 0 else 1
                    #     print(f'this_turn_positive_sum / (-this_turn_negative_sum): {this_turn_ratio}')
                    #     scores[this_turn_negative_indices] *= this_turn_ratio * less_negative.target_ratio


query_scores_indices = []
                    max_turn = 0
                    for i in range(bsz):
                        relevance_scores_indices = torch.nonzero(token_level_query_scores[i]).squeeze()
                        relevance_scores_indices = relevance_scores_indices.tolist()
                        if type(relevance_scores_indices) != list:
                            relevance_scores_indices = [relevance_scores_indices]
                        query_scores_indices.append(relevance_scores_indices)
                        max_turn = max(max_turn, len(relevance_scores_indices))
                    all_turn_ids = []
                    for i in range(bsz):
                        last_turn_ids = None
                        for turn_idx in range(len(query_scores_indices[i])):
                            turn_ids = torch.full((query_scores_indices[i][turn_idx],), turn_idx + 1)
                            if last_turn_ids is not None:
                                turn_ids[:last_turn_ids.shape[0]] = last_turn_ids
                            last_turn_ids = turn_ids
                        turn_ids = torch.full((response_length,), max_turn + 1)
                        if last_turn_ids is not None:
                            turn_ids[:last_turn_ids.shape[0]] = last_turn_ids
                        all_turn_ids.append(turn_ids)
                    all_turn_ids = torch.stack(all_turn_ids) * eos_mask
                    for turn_idx in range(1, max_turn + 2):
                        this_turn_indices = torch.where(all_turn_ids == turn_idx)
                        this_turn_advantage = scores[this_turn_indices]
                        this_turn_positive_mask = this_turn_advantage > 0
                        this_turn_negative_mask = this_turn_advantage < 0
                        this_turn_positive_indices = (this_turn_indices[0][this_turn_positive_mask], this_turn_indices[1][this_turn_positive_mask])
                        this_turn_negative_indices = (this_turn_indices[0][this_turn_negative_mask], this_turn_indices[1][this_turn_negative_mask])
                        this_turn_positive_sum = scores[this_turn_positive_indices].sum()
                        this_turn_negative_sum = scores[this_turn_negative_indices].sum()
                        this_turn_ratio = this_turn_positive_sum / (-this_turn_negative_sum) if this_turn_negative_sum != 0 else 1
                        print(f"Turn {turn_idx} positive/negative ratio: {this_turn_ratio}")
                        scores[this_turn_negative_indices] *= this_turn_ratio * less_negative.target_ratio


                                        # for turn_idx in range(1, max_turn + 2):
                #     this_turn_indices = torch.where(all_turn_ids == turn_idx)
                #     this_turn_advantage = all_advantage[this_turn_indices]
                #     this_turn_positive_mask = this_turn_advantage > 0
                #     this_turn_negative_mask = this_turn_advantage < 0
                #     this_turn_positive_indices = (this_turn_indices[0][this_turn_positive_mask], this_turn_indices[1][this_turn_positive_mask])
                #     this_turn_negative_indices = (this_turn_indices[0][this_turn_negative_mask], this_turn_indices[1][this_turn_negative_mask])
                #     this_turn_positive_sum = all_advantage[this_turn_positive_indices].sum()
                #     this_turn_negative_sum = all_advantage[this_turn_negative_indices].sum()
                #     this_turn_ratio = this_turn_positive_sum / (-this_turn_negative_sum) if this_turn_negative_sum != 0 else 1
                #     print(f"Turn {turn_idx} positive/negative ratio: {this_turn_ratio}")
                #     all_advantage[this_turn_negative_indices] *= this_turn_ratio * less_negative.target_ratio
