import os, glob, time, motor.motor_asyncio, asyncio
from urllib.parse import quote_plus, urlparse, urlunparse
from openai import AzureOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

embedding_model_name = 'embedding01' # OpenAI Studioでデプロイしたモデルの名前
gpt_model_name = 'gpt-4-turbo' # OpenAI Studioでデプロイしたモデルの名前

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

async def process_record(doc):
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
    await collection.replace_one({'_id': doc['_id']}, doc)

async def process_records():
  cursor = collection.find({}, no_cursor_timeout=True)
  tasks = []
  try:
    async for doc in cursor:
        task = asyncio.ensure_future(process_record(doc))
        tasks.append(task)
    await asyncio.gather(*tasks)
  finally:
    await cursor.close()

loop = asyncio.get_event_loop()
loop.run_until_complete(process_records())