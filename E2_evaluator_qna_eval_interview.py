#
import openai
from mlx_lm import load, stream_generate, generate
from rouge_score import rouge_scorer
# client libraries
from utils.ols_client import olsClient
#
from tqdm import tqdm
import pandas as pd
import os, sys, time, datetime

from dotenv import load_dotenv
if not load_dotenv("config.env"):
    print(f"ERROR: Missing config file.")
    sys.exit()

DEBUG = True
MLX_MODEL = None # global

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
# functions for interviewing the models
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
            print(
                f"Previous checkpoint found ({chkpt_fname}). Resumming execution...")
            df = dfN.copy()
        except FileNotFoundError:
            print(
                f"No previous checkpoint ({chkpt_fname}). Initializing a new run....")
            df.to_parquet(chkpt_fname, compression='snappy')
    else:
        df.to_parquet(chkpt_fname, compression='snappy')
    return df

def load_mlx(model):
    # using global to store instance to speedup query loop
    global MLX_MODEL

    if MLX_MODEL is None:
        repo = "instructlab/"+os.getenv(f"{model}_MODEL", "granite-7b-lab")
        mlx_model, mlx_tokenizer = load(repo)
        MLX_MODEL=(mlx_model, mlx_tokenizer)
    return MLX_MODEL

def message_template(q, model):
    # mistral models do not have "system"
    if model.lower().startswith('mistral') is True:
        messages = [
            {
                'role': 'user',
                'content': q.strip()
            }
        ]
    else:
        message=[
            {
                'role': 'system',
                # System instructions for the task
                'content': f'Answer questions truthfully.\n'
            },
            {
                'role': 'user',
                'content': q.strip()
            }
        ]
    return message

def query_llm(q, model):
    """
    send query to model
    """
    ols_path=False
    mlx_path=False

    if model.startswith("OLS_"):
        OLS_API = os.getenv(f"{model}_BASE_URL")
        OLS_TOKEN = os.getenv(f"{model}_TOKEN")
        OLS_PROVIDER = os.getenv(f"{model}_PROVIDER")
        OLS_MODEL = os.getenv(f"{model}_MODEL")
        if OLS_API is None:
            print(f"ERRROR: Cannot find OLS API definition {model}_BASE_URL")
            sys.exit()
        llm_client = olsClient(api_url=OLS_API, token=OLS_TOKEN, llm_provider=OLS_PROVIDER, model=OLS_MODEL)
        print(f"Using OLS PATH Mode\nUrl={OLS_API}\nProvider={OLS_PROVIDER}")
        ols_path=True
    else:
        llm_client = openai.OpenAI(
            base_url=os.getenv(f"{model}_BASE_URL"),
            api_key=os.getenv(f"{model}_API_KEY"),
        )
        model_name = os.getenv(f"{model}_MODEL")
    counter=3   # maximum timeout cycles for question
    while True:
        try:
            if ols_path is True:
                response = llm_client.query(q.strip())
                print(f"DEBUG: {model} response: {response}")
            elif model.lower().startswith('granite'):
                mlx_model, mlx_tokenizer = load_mlx(model)
                response = generate(model=mlx_model, tokenizer=mlx_tokenizer,
                                    prompt=q.strip(), max_tokens=1000)
                print(f"DEBUG: MLX response: {response}")
                mlx_path=True
            else:
                response = llm_client.chat.completions.create(
                    model=model_name,
                    messages=message_template(q, model),
                    # response_format={ "type": "json_object" },
                    temperature=0.3,
                    top_p=0.1,
                    timeout=30,
                    stream=False,
                )
                print(
                    f"DEBUG: {model} response: {response.choices[0].message.content}")
            if ols_path is True or mlx_path is True:
                answer=response
            else:
                answer=response.choices[0].message.content
            if DEBUG is True:
                print(f"DEBUG. Answer = {answer}")
            break
        except openai.APITimeoutError:
            counter-=1
            if counter <= 0:
                print(f"ERROR: Too many timeouts. Ignoring question: {q}")
                answer=pd.NA
                break
            else:
                print(f"WARNING: Backend timeout. Retrying up to {counter} times.")
                with open("errors_q.txt", 'a') as err_file:
                    err_file.write(f"[{model}] "+q+"\n")
                err_file.close()
                time.sleep(3)
    del(llm_client)
    return answer

def interview_model(df, model, quantity=5, chkpt=True):
    """
    process the top {quantity} questions in data frame
    """
    # if invalid number of question, use the full set
    if quantity <= 0:
        quantity = df.shape[0]
    
    top_n = df[['Question']].head(quantity)

    if model not in df.columns:
        # make sure the column exist for continuation logic
        df[model]=pd.NA

    chkpt_indx=0
    for indx, row in tqdm(top_n.iterrows(), desc=f"Interviewing {model} Q={quantity}"):
        # only invoke the llm if there is no answer with this model
        try:
            if df.isnull().loc[indx, model].sum() > 0:
                df.loc[indx, model] = query_llm(row['Question'], model)
                chkpt_indx+=1
                # if enabled, do checkpoint every 5 new entries
                if chkpt and (chkpt_indx % 5) == 0:
                    print(f" Checkpoint @ {datetime.datetime.now()}")
                    checkpoint(df)
            else:
                continue
        except Exception as e:
            print(
                f"ERROR: {e}\n df.loc results: {df.isnull().loc[indx, model]} with count={df.isnull().loc[indx, model].sum()}")
            sys.exit()
    return df

def interview_all(df, q_num=30):
    """
    handle interview for all models
    """
    for m in MODELS:
        df = interview_model(df, m, q_num)
        checkpoint(df)
    return df

####################################################################################
# main
####################################################################################
if __name__ == '__main__':
    """
    """
    qna_df = pd.DataFrame()    
    qna_df = checkpoint(qna_df)

    if qna_df.shape[0] == 0:
        print(f"ERROR: Empty QNA Eval Pool")
        sys.exit()

    if DEBUG is True:
        print(f"Initialized DataFrame shape={qna_df.shape}\nColumns: {list(qna_df.columns)}")

    # interview models
    qna_df = interview_all(qna_df,1000)
    
    print(f"DataFrame shape={qna_df.shape} Columns: {qna_df.columns}")

    checkpoint(qna_df, "interview-checkpoint.parquet")

####################################################################################
# END OF FLE
####################################################################################