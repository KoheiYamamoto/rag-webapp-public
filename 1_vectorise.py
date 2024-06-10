import os
import glob
import time
from urllib.parse import quote_plus, urlparse, urlunparse

import asyncio
import motor.motor_asyncio
from openai import AzureOpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()

#import nest_asyncio
#nest_asyncio.apply()

# MongoDBの設定
mongo_conn_str = os.getenv('COSMOS_CONNECTION_STRING') # MongoDBの接続文字列
url_parts = list(urlparse(mongo_conn_str)) # 接続文字列を分解
escaped_username = quote_plus(os.getenv('COSMOS_USERNAME')) # エスケープ：ユーザー名をURLエンコード
escaped_password = quote_plus(os.getenv('COSMOS_PASSWORD')) # エスケープ：パスワードをURLエンコード
url_parts[1] = url_parts[1].replace('<user>:<password>', f'{escaped_username}:{escaped_password}') # ユーザー名とパスワードを置換
mongo_conn_str = urlunparse(url_parts) # 接続文字列を再構築

db_name = "db1"  # データベース名
collection_name = "coll_holtest"  # コレクション名

embedding_model_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_VECTORISE') # OpenAI Studioでデプロイしたモデルの名前

# Azure OpenAIのクライアントを生成
    
client = AzureOpenAI(
    api_key=os.getenv('AZURE_OPENAI_API_KEY_VECTORISE'),  
    api_version="2023-12-01-preview",
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT_VECTORISE')
)

# MongoDB Clientを生成
mongoclient = motor.motor_asyncio.AsyncIOMotorClient(mongo_conn_str)
db = mongoclient[db_name]
collection = db[collection_name]

# ファイルを読みだしてEmbeddingを取得してMongoDBに保存する非同期関数
async def store_embedding(cnt,filename):

    with open(filename, 'r',encoding='utf8') as data:
        text = data.read().replace('\n', '')
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000)
    chunks = splitter.split_text(text)
    
    for num in range(len(chunks)):
    
        try:
            vectors = client.embeddings.create(model=embedding_model_name,input=chunks[num]).data[0].embedding
        
        except Exception as e:
            print (f"Error when calling embeddings.create():[{e}]")

        collection.insert_one({"name":filename,"num":num,"vectors":vectors,"text":chunks[num]})

        print(f"{cnt} : {filename} - Chunk[{num+1}/{len(chunks)}] Inserted ")


# メインの非同期イベントループ
async def main():

    # コレクションをクリア
    await collection.drop()
    print("Documents Droped : count = " + str(await collection.count_documents({})) )

    #ベクトルインデックス定義
    await db.command(
      {
        "createIndexes": collection_name,
        "indexes": [
          {
            "name": "idx_vectors",
            "key": {
              "vectors": "cosmosSearch"
            },
            "cosmosSearchOptions": {
              "kind": 'vector-ivf',
              "numLists":100,
              "similarity": 'COS',
              "dimensions": 1536
            }
          }
        ]
      }
    )

    #ベクトルインデックス定義
    await db.command(
      {
        "createIndexes": collection_name,
        "indexes": [
          {
            "name": "idx_summary_vectors",
            "key": {
              "summary_vectors": "cosmosSearch"
            },
            "cosmosSearchOptions": {
              "kind": 'vector-ivf',
              "numLists":100,
              "similarity": 'COS',
              "dimensions": 1536
            }
          }
        ]
      }
    )

    
    # ファイル名をstore_embeddingに引き渡して実行
    files = glob.glob(os.getcwd() + '/test1000/*.txt') # カレントディレクトリのパスを取得し、test1000フォルダ内のすべてのtxtファイルを取得
    # files = glob.glob('/home/xxxx/test1000/*.txt')    
    files.sort()
    files = files[0:100] # 100ファイルのみ

    cnt = 0
    for file in files :
        cnt = cnt + 1 
        await store_embedding(cnt,file)

    time.sleep(5)

# main()を呼び出す

if __name__ == '__main__':
    asyncio.run(main())