# Create the Q&A eval pool

import pandas as pd
from tqdm import tqdm
from utils import db_conn
import os, sys

from dotenv import load_dotenv
if not load_dotenv("config.env"):
    print(f"ERROR: Missing config file.")
    sys.exit()

DEBUG=False

def qna_pool(conn, title='', order_by='RANDOM()', num=100):
    """
    When title='' it will select from all topics
    """
    cur = conn.cursor()
    cur.execute(f"""
        SELECT
            id,
            question,
            answer,
            doc_source,
            doc_title,
            doc_page_num,
            ocp_version,
            h_validated,
            m_validated
        FROM
            qna_pool_hf
        WHERE
            doc_title ~ '{title}'
        ORDER BY
            {order_by}
        LIMIT
            {num}
        ;
    """)
    content = []
    rows = cur.fetchall()
    for id, q, a, d_src, d_title, d_page, ocp_ver, h_valid, m_valid in tqdm(rows):
        content.append(
            {
                "ID": id,
                "Question": q.strip(),
                "Answer": a.strip(),
                "doc_source": d_src.strip(),
                "doc_title": d_title.strip(),
                "doc_page": d_page,
                "ocp_version": ocp_ver.strip(),
                "human_validated": bool(h_valid),
                "model_validated": bool(m_valid)
            }
        )
    cur.close()
    return content


if __name__ == '__main__':
    """
    """
    conn = db_conn.conn()

    cur = conn.cursor()
    # fetch unique titles
    cur.execute("""
        SELECT DISTINCT
            doc_title
        FROM
            qna_pool_hf
        ;
    """)
    titles = []
    _titles = cur.fetchall()
    for i in _titles:
        titles.append(i[0].strip())
    
    # print title
    print(f"Topics: {titles}")

    frames = []
    for t in tqdm(titles):
        print(f"""\n{'#'*40}""")
        print(f"Processing: {t}")
        # select 30 random Q&A pairs per topic
        content = qna_pool(conn, title=t, order_by='RANDOM()', num=30)
        if DEBUG is True:
            for entry in content:
                print(f"{entry['Question']}")
        df = pd.DataFrame(content).set_index('ID')
        if DEBUG is True:
            print(f"""{df.head()}\n{df.dtypes}""")
        frames.append(df)  # append to the list of dataframes

    df_full = pd.concat(frames).sort_index(ascending=True)
    print(df_full.head(), df_full.shape)

    QNA_EVAL_POOL=os.environ.get("QNA_EVAL_POOL", "unk_qna_eval_pool.parquet")
    print(f"Writing {QNA_EVAL_POOL}...")
    df_full.to_parquet(QNA_EVAL_POOL)

    ##
    # Close DB connection
    cur.close()
    conn.close()

####################################################################################
# END OF FLE
####################################################################################
