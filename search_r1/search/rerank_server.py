import argparse
from collections import defaultdict
from typing import Optional
from dataclasses import dataclass, field

from sentence_transformers import CrossEncoder, SentenceTransformer
import torch
from transformers import HfArgumentParser, AutoModel, AutoTokenizer, AutoModelForCausalLM
import numpy as np

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class MultiGpuCrossReranker:
    def __init__(self, model_name_or_path, batch_size=32):
        self.batch_size = batch_size
        self.num_gpus = torch.cuda.device_count()
        self.models = []

        if 'Qwen3-Reranker' in model_name_or_path:
            self.model_type = "CausalLM"
        elif 'ms-marco-MiniLM-L12-v2' in model_name_or_path:
            self.model_type = "CrossEncoder"
        else:
            raise ValueError(f"Model {model_name_or_path} not supported")

        # 预加载到每个 GPU
        for device_id in range(self.num_gpus):
            print(f"Loading model on cuda:{device_id}")
            if self.model_type == "CausalLM":
                model = AutoModelForCausalLM.from_pretrained(model_name_or_path).eval()
                model = model.to(f"cuda:{device_id}")
            elif self.model_type == "CrossEncoder":
                model = CrossEncoder(model_name_or_path, device=f"cuda:{device_id}")
            self.models.append(model)
        
        if 'Qwen3-Reranker' in model_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='left')
            self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
            self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
            self.max_length = 8192

            # prefix_qd = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
            prefix_qad = "<|im_start|>system\nJudge whether a document can help correctly answer a given question based on the question and its corresponding answer. The question may be a complex multi-hop question, and a helpful document does not necessarily contain the answer but should be logically related to both the question and the answer. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
            prefix_ad = "<|im_start|>system\nJudge whether the document is related to the phrase. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            # self.prefix_qd_tokens = self.tokenizer.encode(prefix_qd, add_special_tokens=False)
            self.prefix_qad_tokens = self.tokenizer.encode(prefix_qad, add_special_tokens=False)
            self.prefix_ad_tokens = self.tokenizer.encode(prefix_ad, add_special_tokens=False)
            self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

    def CausalLM_predict(self, model, pairs: list[tuple], batch_size=32):

        def format_instruction(instruction, query, doc):
            if instruction is None:
                instruction = 'Given a web search query, retrieve relevant passages that answer the query'
            output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
            return output
        
        def format_qad(query, doc, answer):
            if query is None:  
                output = "<Phrase>: {answer}\n<Document>: {doc}".format(answer=answer, doc=doc)
            else:
                output = "<Query>: {query}\n<Answer>: {answer}\n<Document>: {doc}".format(query=query, answer=answer, doc=doc)
            return output
        
        def process_inputs(pairs, prefix_type="ad"):
            if '<Query>' in pairs[0]:
                prefix_tokens = self.prefix_qad_tokens
            elif '<Phrase>' in pairs[0]:
                prefix_tokens = self.prefix_ad_tokens
            else:
                raise ValueError(f"Format not supported")
            inputs = self.tokenizer(
                pairs, padding=False, truncation='longest_first',
                return_attention_mask=False, max_length=self.max_length - len(prefix_tokens) - len(self.suffix_tokens)
            )
            for i, ele in enumerate(inputs['input_ids']):
                inputs['input_ids'][i] = prefix_tokens + ele + self.suffix_tokens
            inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
            for key in inputs:
                inputs[key] = inputs[key].to(model.device)
            return inputs

        @torch.no_grad()
        def compute_logits(inputs, micro_batch_size=32):
            input_len = inputs["input_ids"].shape[0]
            print(f"input_len: {input_len}")
            scores = []

            # 按 micro_batch_size 分批处理
            for start in range(0, input_len, micro_batch_size):
                end = min(start + micro_batch_size, input_len)
                micro_inputs = {k: v[start:end] for k, v in inputs.items()}

                with torch.no_grad():
                    batch_scores = model(**micro_inputs).logits[:, -1, :]
                    true_vector = batch_scores[:, self.token_true_id]
                    false_vector = batch_scores[:, self.token_false_id]
                    batch_scores = torch.stack([false_vector, true_vector], dim=1)
                    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                    micro_scores = batch_scores[:, 1].exp().tolist()
                    scores.extend(micro_scores)
                torch.cuda.empty_cache()

            return scores

        if len(pairs[0]) == 2:
            task = 'Given a web search query, retrieve relevant passages that answer the query'
            pairs = [format_instruction(task, query, doc) for query, doc in pairs]
        elif len(pairs[0]) == 3:
            pairs = [format_qad(query, doc, answer) for query, doc, answer in pairs]

        inputs = process_inputs(pairs)
        scores = compute_logits(inputs, batch_size)
        return scores

    def predict(self, pairs: list[tuple]):
        # 切分数据到多个 GPU
        chunks = [list(chunk) for chunk in np.array_split(pairs, self.num_gpus)]
        results = []

        def worker(device_id, chunk):
            if len(chunk) == 0:
                return []
            if len(chunk[0]) == 2:
                chunk = [(str(q), str(d)) for q, d in chunk]
            else:
                if chunk[0][0] is None:
                    chunk = [(None, str(d), str(a)) for q, d, a in chunk]
                else:
                    chunk = [(str(q), str(d), str(a)) for q, d, a in chunk]
            model = self.models[device_id]
            if self.model_type == "CrossEncoder":
                scores = model.predict(chunk, batch_size=self.batch_size)
                torch.cuda.empty_cache()
                scores = scores.tolist()
            elif self.model_type == "CausalLM":
                scores = self.CausalLM_predict(model, chunk, self.batch_size)
            return scores

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            futures = [executor.submit(worker, i, chunk) for i, chunk in enumerate(chunks)]
            for f in futures:
                results.extend(f.result())

        return results

class CrossReranker:
    def __init__(self, multi_gpu_model):
        self.multi_gpu_model = multi_gpu_model

    def _passage_to_string(self, doc_item):
        if 'str' in doc_item:
            return doc_item['str']
        if "document" not in doc_item:
            content = doc_item['contents']
        else:
            content = doc_item['document']['contents']
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])

        return f"(Title: {title}) {text}"

    def rerank(self, 
               queries: list[str], 
               documents: list[list[dict]],
               not_sort = False,
               answers = None):
        """
        Assume documents is a list of list of dicts, where each dict is a document with keys "id" and "contents".
        This asumption is made to be consistent with the output of the retrieval server.
        """ 
        assert len(queries) == len(documents)

        pairs = []
        qids = []
        for qid, query in enumerate(queries):
            for doc_item in documents[qid]:
                doc = self._passage_to_string(doc_item)
                if answers is not None:
                    pairs.append((query, doc, answers[qid]))
                else:
                    pairs.append((query, doc))
                qids.append(qid)

        scores = self._predict(pairs)
        query_to_doc_scores = defaultdict(list)

        assert len(scores) == len(pairs) == len(qids)
        for i in range(len(pairs)):
            if len(pairs[i]) == 2:
                query, doc = pairs[i]
            else:
                query, doc, answer = pairs[i]
            score = scores[i] 
            qid = qids[i]
            query_to_doc_scores[qid].append((doc, score))

        sorted_query_to_doc_scores = {}
        for query, doc_scores in query_to_doc_scores.items():
            sorted_query_to_doc_scores[query] = sorted(doc_scores, key=lambda x: x[1], reverse=True)

        if not_sort:
            return query_to_doc_scores
        return sorted_query_to_doc_scores

    def _predict(self, pairs: list[tuple[str, str]]):
        return self.multi_gpu_model.predict(pairs)

    @classmethod
    def load(cls, model_name_or_path, **kwargs):
        batch_size = kwargs.get("batch_size", 32)
        multi_gpu_model = MultiGpuCrossReranker(model_name_or_path, batch_size=batch_size)
        return cls(multi_gpu_model)

class MultiGpuBiEncoder:
    def __init__(self, model_name_or_path, batch_size=32):
        self.batch_size = batch_size
        self.num_gpus = torch.cuda.device_count()
        self.models = []

        # 预加载到每个 GPU
        for device_id in range(self.num_gpus):
            print(f"Loading model on cuda:{device_id}")
            model = SentenceTransformer(model_name_or_path, device=f"cuda:{device_id}")
            self.models.append(model)

    def predict(self, queries: list[str], docs: list[list[str]]):
        query_doc_dict = []
        for idx in range(len(queries)):
            query_doc_dict.append({"query": queries[idx], "docs": docs[idx]})
        chunks = [list(chunk) for chunk in np.array_split(query_doc_dict, self.num_gpus)]
        results = []

        def worker(device_id, chunk):
            model = self.models[device_id]
            queries = [item["query"] for item in chunk]
            docs = []
            for item in chunk:
                docs.extend(item["docs"])
            query_embeddings = model.encode(queries, prompt_name="query")
            doc_embeddings = model.encode(docs)
            scores = []
            for query_idx in range(len(queries)):
                the_query_embedding = query_embeddings[query_idx]
                the_doc_embeddings = doc_embeddings[query_idx * len(item["docs"]):(query_idx + 1) * len(item["docs"])]
                similarity = model.similarity(the_query_embedding, the_doc_embeddings)
                scores.append(similarity.tolist()[0])
            return scores

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            futures = [executor.submit(worker, i, chunk) for i, chunk in enumerate(chunks)]
            for f in futures:
                results.extend(f.result())

        return results
            

class BiReranker:
    def __init__(self, multi_gpu_model, batch_size=32):
        self.multi_gpu_model = multi_gpu_model
        self.batch_size = batch_size

    def _passage_to_string(self, doc_item):
        if 'str' in doc_item:
            return doc_item['str']
        if "document" not in doc_item:
            content = doc_item['contents']
        else:
            content = doc_item['document']['contents']
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])

        return f"(Title: {title}) {text}"

    def rerank(self, 
        queries: list[str], 
        documents: list[list[dict]]): 

        assert len(queries) == len(documents)

        docs = []
        for doc_items in documents:
            this_query_docs = []
            for doc_item in doc_items:
                this_query_docs.append(self._passage_to_string(doc_item))
            docs.append(this_query_docs)

        scores = self._predict(queries, docs)

        query_to_doc_scores = defaultdict(list)
        for query_idx in range(len(queries)):
            for doc_idx in range(len(docs[query_idx])):
                query_to_doc_scores[query_idx].append((docs[query_idx][doc_idx], scores[query_idx][doc_idx]))
        
        sorted_query_to_doc_scores = {}
        for query, doc_scores in query_to_doc_scores.items():
            sorted_query_to_doc_scores[query] = sorted(doc_scores, key=lambda x: x[1], reverse=True)

        return sorted_query_to_doc_scores
        
    def _predict(self, queries: list[str], docs: list[list[str]]):
        return self.multi_gpu_model.predict(queries, docs)
    
    @classmethod
    def load(cls, model_name_or_path, **kwargs):
        batch_size = kwargs.get("batch_size", 32)
        multi_gpu_model = MultiGpuBiEncoder(model_name_or_path, batch_size=batch_size)
        return cls(multi_gpu_model, batch_size=batch_size)

class RerankRequest(BaseModel):
    queries: Optional[list[str]] = None
    documents: list[list[dict]]
    rerank_topk: Optional[int] = None
    return_scores: bool = False
    not_sort: Optional[bool] = False
    answers: Optional[list[str]] = None


@dataclass 
class RerankerArguments:
    max_length: int = field(default=512)
    rerank_topk: int = field(default=3)
    rerank_model_name_or_path: str = field(default="cross-encoder/ms-marco-MiniLM-L12-v2")
    batch_size: int = field(default=32)
    reranker_type: str = field(default="sentence_transformer")

def get_reranker(config):
    if config.reranker_type == "sentence_transformer":
        if 'ms-marco-MiniLM-L12-v2' in config.rerank_model_name_or_path or 'Qwen3-Reranker' in config.rerank_model_name_or_path:
            return CrossReranker.load(
                config.rerank_model_name_or_path,
                batch_size=config.batch_size
            )
    else:
        raise ValueError(f"Unknown reranker type: {config.reranker_type}")


app = FastAPI()

@app.post("/rerank")
def rerank_endpoint(request: RerankRequest):
    """
    Endpoint that accepts queries and performs retrieval.
    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "documents": [[doc_item_1, ..., doc_item_k], [doc_item_1, ..., doc_item_k]],
      "rerank_topk": 3,
      "return_scores": true
    }
    """
    if not request.rerank_topk:
        request.rerank_topk = config.rerank_topk  # fallback to default

    # Perform batch re reranking
    # doc_scores already sorted by score
    if request.answers:
        if not request.queries:
            request.queries = [None for _ in range(len(request.documents))]
        query_to_doc_scores = reranker.rerank(request.queries, request.documents, request.not_sort, request.answers) 
    else:
        query_to_doc_scores = reranker.rerank(request.queries, request.documents, request.not_sort) 

    # Format response 
    resp = []
    for _, doc_scores in query_to_doc_scores.items():
        doc_scores = doc_scores[:request.rerank_topk]
        if request.return_scores:
            combined = [] 
            for doc, score in doc_scores:
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            resp.append([doc for doc, _ in doc_scores])
    return {"result": resp}


if __name__ == "__main__":
    
    # 1) Build a config (could also parse from arguments).
    #    In real usage, you'd parse your CLI arguments or environment variables.
    parser = HfArgumentParser((RerankerArguments))
    config = parser.parse_args_into_dataclasses()[0]

    # 2) Instantiate a global retriever so it is loaded once and reused.
    reranker = get_reranker(config)
    
    # 3) Launch the server. By default, it listens on http://127.0.0.1:8000
    uvicorn.run(app, host="0.0.0.0", port=6980)
