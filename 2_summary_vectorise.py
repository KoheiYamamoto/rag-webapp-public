import threading
import os, glob, time, motor.motor_asyncio, asyncio
from urllib.parse import quote_plus, urlparse, urlunparse
from openai import AzureOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

embedding_model_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_VECTORISE')
gpt_model_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_SUMMARISE') 

def setup_clients():
  """
  Set up Azure OpenAI client and MongoDB client.
  Returns:
    Tuple: A tuple containing the Azure OpenAI client and MongoDB collection.
  """
  # Azure OpenAIのクライアントを生成    
  client = AzureOpenAI(
    api_key=os.getenv('AZURE_OPENAI_API_KEY_SUMMARISE'),  
    api_version="2023-12-01-preview",
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT_SUMMARISE')
  )
  # MongoDBの設定
  mongo_conn_str = os.getenv('COSMOS_CONNECTION_STRING') # MongoDBの接続文字列
  url_parts = list(urlparse(mongo_conn_str)) # 接続文字列を分解
  escaped_username = quote_plus(os.getenv('COSMOS_USERNAME')) # エスケープ：ユーザー名をURLエンコード
  escaped_password = quote_plus(os.getenv('COSMOS_PASSWORD')) # エスケープ：パスワードをURLエンコード
  url_parts[1] = url_parts[1].replace('<user>:<password>', f'{escaped_username}:{escaped_password}') # ユーザー名とパスワードを置換
  mongo_conn_str = urlunparse(url_parts) # 接続文字列を再構築

  db_name = "db1"  # データベース名を設定してください
  collection_name = "coll_holtest"  # コレクション名を設定してください

  # MongoDB Clientを生成
  mongoclient = motor.motor_asyncio.AsyncIOMotorClient(mongo_conn_str)
  db = mongoclient[db_name]
  collection = db[collection_name]
  return client, db, collection

# クライアントをセットアップ
client, db, collection = setup_clients()

# mongo db からデータを取得
async def get_data():
  """
  Get data from MongoDB.
  Returns:
    List: A list of documents from the MongoDB collection.
  """
  cursor = collection.find({})
  return await cursor.to_list(None)

def process_record(doc):
    print(f"--- Processing record: {doc['_id']} --- ")
    # もし、要約がすでに存在している場合は、スキップ
    if 'summary' in doc:
        print("Summary already exists. Skipping.")
        return
    try: 
      summary = client.chat.completions.create(
          model=gpt_model_name,
          messages=[
              {"role": "user", "content": "余分な情報を取り除いて要約してください、ただし、重要だと思う情報はできるだけ残してください。また、改行などの構成は工夫しないでください。パッセージのみの出力をしてください/n" + doc['text']}
          ]
      )
      summary = summary.choices[0].message.content
      if summary == "": # 要約が空の場合は、元のテキストを使う
        summary = doc['text']
    except Exception as e: # エラーが起きた場合は、元のテキストを使う
      print(f"Error occurred: {e}. Using original text.")
      summary = doc['text']    
    print(f"Summary: {summary}")
    # 要約のベクトル化
    vectors = client.embeddings.create(model=embedding_model_name, input=summary).data[0].embedding
    # レコードの更新
    doc['summary'] = summary
    doc['summary_vectors'] = vectors
    collection.replace_one({'_id': doc['_id']}, doc)

# データを表示
async def dispatch_thread():
    """
    Print data from MongoDB.
    """
    data = await get_data()

    threads = []
    for doc in data:
        t = threading.Thread(target=process_record, args=(doc,))
        t.start()
        threads.append(t)

    # Wait for all threads to complete
    for t in threads:
        t.join()

if __name__ == "__main__":
    asyncio.run(dispatch_thread())