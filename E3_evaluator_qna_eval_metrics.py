#
from rouge_score import rouge_scorer
# metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate import meteor
import nltk
nltk.download('wordnet')
#
from tqdm import tqdm
import pandas as pd
import os, sys, json, time, datetime

from dotenv import load_dotenv
if not load_dotenv("config.env"):
    print(f"ERROR: Missing config file.")
    sys.exit()

DEBUG = True

####################################################################################
# load globals vars
if not load_dotenv("config.env"):
    print(f"ERROR: Missing config file.")
    sys.exit()

MODELS = [
    'GRANITE7B',
    'OLS_GRANITE',
    'OPENAI',
    'OLS_OAI',
]

####################################################################################
#
####################################################################################
def checkpoint(df, fname=None):
    """
    load or create a checkpoint
    """
    chkpt_fname = "eval-auto-checkpoint.parquet" if fname is None else fname

    if df.shape[0] == 0:
        # empty df, try to load previous
        try:
            dfN = pd.read_parquet(chkpt_fname)
            print(f"Previous checkpoint found ({chkpt_fname}). Resumming execution...")
            df = dfN.copy()
        except FileNotFoundError:
            print(f"No previous checkpoint ({chkpt_fname}). Initializing a new run....")
            df.to_parquet(chkpt_fname, compression='snappy')
    else:
        df.to_parquet(chkpt_fname, compression='snappy')
    return df

##
# Scores
##
def score_info():
    """
    ___ABOUT SCORES___

    GLEU   : Evaluation metric to estimate fluency. Compares model generated text and human generated text. (higher the better, 0 to 1)
    
    METEOR : Precision-based metric to measure quality of generated text. Allows synonyms and stemmed words to be matched with the reference word.

    ROUGE  : Recall focused metrics. Compares quality of generated to reference text. How many words a model recall?
             ROUGE-1 scores are excellent around 0.5, with scores above 0.5 considered good and 0.4 to 0.5 moderate
             ROUGE-L scores are good around 0.4 and low at 0.3 to 0.4.
    """
    print(f"\n{'#'*20}\n\n{score_info.__doc__}")

def score_gleu(df, models=MODELS):
    """

    """
    for m in tqdm(models):
        if f"{m}@gleu" not in df.columns:
            df[f"{m}@gleu"] = pd.NA

        for indx, row in df.iterrows():
            colname = f"{m}@gleu"
            # only calculate if there is no answer with this model
            if bool(df.isnull().loc[indx, colname]) is True:
                q_truth = row['Answer'].strip().split()
                q_hyp00 = row[m].strip().split()
                df.loc[indx, colname] = sentence_gleu([q_truth], q_hyp00)
        
    #print(df.head(10),list(df.columns))
    print(f"\n{'#'*20} GLEU SCORE\n\n{'Model':>23} " +
          f"{'P95':>12} " +
          f"{'P99':>12} "
          )
    for m in models:
        colname = f"{m}@gleu"
        print(f"{m:>23} " +
              f"{df[colname].quantile(0.95):12.3f} " +
              f"{df[colname].quantile(0.99):12.3f}"
              )

    return df

def score_meteor(df, models=MODELS):
    """

    """
    for m in tqdm(models):
        if f"{m}@meteor" not in df.columns:
            df[f"{m}@meteor"] = pd.NA

        for indx, row in df.iterrows():
            colname = f"{m}@meteor"
            # only calculate if there is no answer with this model
            if bool(df.isnull().loc[indx, colname]) is True:
                q_truth = row['Answer'].strip().split()
                q_hyp00 = row[m].strip().split()
                df.loc[indx, colname] = meteor([q_truth], q_hyp00)

    # print(df.head(10),list(df.columns))
    print(f"\n{'#'*20} METEOR SCORE \n\n{'Model':>23} " +
            f"{'P95':>12} " +
            f"{'P99':>12} "
            )
    for m in models:
        colname = f"{m}@meteor"
        print(f"{m:>23} " +
              f"{df[colname].quantile(0.95):12.3f} " +
              f"{df[colname].quantile(0.99):12.3f}"
              )

    return df

def score_rouge(df, models=MODELS):
    """
    Calculate Rouge score
    """
    scorer = rouge_scorer.RougeScorer(['rouge5', 'rougeL'], use_stemmer=True)
    for m in tqdm(models):
        if f"{m}@rouge" not in df.columns:
            df[f"{m}@rouge"] = pd.NA

        for indx, row in df.iterrows():
            colname = f"{m}@rouge"
            # only calculate if there is no answer with this model
            if bool(df.isnull().loc[indx, colname]) is True:
                q_truth = row['Answer'].strip()
                q_hyp00 = row[m].strip()
                #df.loc[indx, colname] = scorer.score(q_truth, q_hyp00)
                r_scores = scorer.score(q_truth, q_hyp00)
                r1_p, r1_r, r1_f = r_scores['rouge5']
                rL_p, rL_r, rL_f = r_scores['rougeL']

                df.loc[indx, colname+"N:precision"] = r1_p
                df.loc[indx, colname+"N:recall"] = r1_r
                df.loc[indx, colname+"N:fmeasure"] = r1_f

                df.loc[indx, colname+"L:precision"] = rL_p
                df.loc[indx, colname+"L:recall"] = rL_r
                df.loc[indx, colname+"L:fmeasure"] = rL_f

    # print(df.head(10),list(df.columns))
    print(f"\n{'#'*20} ROUGE SCORE")
    for m in models:
        colname = f"{m}@rouge"
        print(f"\n{m:>23} " +
              f"{'R5-precision':>12} " +
              f"{'R5-recall':>12} " +
              f"{'R5-fmeasure':>12} " +
              f"{'RL-precision':>12} " +
              f"{'RL-recall':>12} " +
              f"{'RL-fmeasure':>12} "
              )
        print(f"{'p95':>23} " +
              f"{df[f'{colname}N:precision'].quantile(0.95):12.3f} " +
              f"{df[f'{colname}N:recall'].quantile(0.95):12.3f} " +
              f"{df[f'{colname}N:fmeasure'].quantile(0.95):12.3f} " +
              f"{df[f'{colname}L:precision'].quantile(0.95):12.3f} " +
              f"{df[f'{colname}L:recall'].quantile(0.95):12.3f} " +
              f"{df[f'{colname}L:fmeasure'].quantile(0.95):12.3f} "
              )
        print(f"{'p99':>23} " +
              f"{df[f'{colname}N:precision'].quantile(0.99):12.3f} " +
              f"{df[f'{colname}N:recall'].quantile(0.99):12.3f} " +
              f"{df[f'{colname}N:fmeasure'].quantile(0.99):12.3f} " +
              f"{df[f'{colname}L:precision'].quantile(0.99):12.3f} " +
              f"{df[f'{colname}L:recall'].quantile(0.99):12.3f} " +
              f"{df[f'{colname}L:fmeasure'].quantile(0.99):12.3f} "
              )
    return df


####################################################################################
# main
####################################################################################
if __name__ == '__main__':
    """
    """
    EVALS_FILE = "eval_scores_checkpoint.parquet"

    qna_df = checkpoint(pd.DataFrame())

    if qna_df.shape[0] == 0:
        print(f"ERROR: Empty QNA Eval Pool")
        sys.exit()

    if DEBUG is True:
        print(f"Initialized DataFrame shape={qna_df.shape}\nColumns: {list(qna_df.columns)}")


    # read existing model score or create a new one
    df_scores = checkpoint(pd.DataFrame(), "eval_scores_checkpoint.parquet")
    if df_scores.shape[0] == 0:
        # if no previous checkpoint seed a new one
        df_scores = qna_df.copy().reset_index()
        df_scores = df_scores[['Question', 'Answer', 'doc_title',
                            'GRANITE7B', 'OLS_GRANITE',
                            'OPENAI', 'OLS_OAI']]
        df_scores = checkpoint(df_scores, "eval_scores_checkpoint.parquet")

    if DEBUG is True:
        print(
            f"Scores DataFrame shape={df_scores.shape}\nColumns: {list(df_scores.columns)}")

    # prepend self answer to calculate theoretical best score
    if 'Answer' not in MODELS:
        MODELS.insert(0, 'Answer')

    # All scores
    score_info()
    df_scores = score_gleu(df_scores, MODELS)
    df_scores = score_meteor(df_scores, MODELS)
    df_scores = score_rouge(df_scores, MODELS)

    print(f"DEBUG: After scores {df_scores.shape}")

    # per topic scores
    # for topic in df_scores['doc_title'].unique():
    #     df_topic = df_scores.query(f'doc_title in {[topic]}')
    #     print("Updated sub-table", topic)
    #     score_rouge(df_topic, MODELS)

    # print(df_scores.columns)
    # #print(df_scores.head(5))
    checkpoint(df_scores, "eval_scores_checkpoint.parquet")

####################################################################################
# END OF FLE
####################################################################################