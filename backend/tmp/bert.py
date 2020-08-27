import spacy
import multiprocessing
import logging
import torch
import pandas as pd
from backend.lib.multiprocessor import job_pool
from backend.tmp.bert_job import job

multiprocessing.set_start_method('spawn', force=True)

is_using_gpu = spacy.prefer_gpu()
if is_using_gpu:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
torch.set_num_threads(1)
nlp_bert = spacy.load("en_trf_bertbaseuncased_lg")
logging.basicConfig(level=logging.DEBUG)


def main():
    job_number = 4
    prevalences = pd.read_csv('data/prevalences.csv')
    df = pd.DataFrame(list(zip(prevalences['Word'][0:200], prevalences['Word'][200:400])), columns=['w1', 'w2'])
    jobs = job_pool(df, job_number, job, model=nlp_bert)
    df = pd.concat(jobs)
    print(df)


if __name__ == "__main__":
    main()
