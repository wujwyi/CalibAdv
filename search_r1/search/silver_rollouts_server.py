import argparse
from collections import defaultdict
from typing import Optional
from dataclasses import dataclass, field
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import json
from typing import List


class ProviderRequest(BaseModel):
    question_idx: List[int]

def get_provider(silver_rollouts_path, max_score_path):
    with open(silver_rollouts_path, "r") as f:
        silver_rollouts = json.load(f)
    with open(max_score_path, "r") as f:
        max_score = json.load(f)
    return silver_rollouts, max_score


app = FastAPI()

@app.post("/get_silver_rollouts")
def provider_endpoint(request: ProviderRequest):
    id2rollouts = {}
    id2maxscore = {}
    for idx in request.question_idx:
        if str(idx) in provider:
            id2rollouts[idx] = provider[str(idx)]
            id2maxscore[idx] = max_score[str(idx)]
    return id2rollouts, id2maxscore


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # retriever
    parser.add_argument("--silver_rollouts_path", type=str, default="rollout/silver_rollouts.json")
    parser.add_argument("--max_score_path", type=str, default="rollout/max_score.json")
    args = parser.parse_args()

    # 2) Instantiate a global retriever so it is loaded once and reused.
    provider, max_score = get_provider(args.silver_rollouts_path, args.max_score_path)
    
    # 3) Launch the server. By default, it listens on http://127.0.0.1:8000
    uvicorn.run(app, host="0.0.0.0", port=6000)
