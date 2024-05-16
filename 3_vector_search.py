import os, motor.motor_asyncio, argparse
from urllib.parse import quote_plus, urlparse, urlunparse
from openai import AzureOpenAI
import asyncio
from dotenv import load_dotenv
load_dotenv()

# Azure OpenAIの設定
openai_key = os.getenv('AZURE_OPENAI_API_KEY_VECTORISE')
openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT_VECTORISE')
openai = AzureOpenAI(
    azure_endpoint=openai_endpoint,
    api_version='2023-12-01-preview',
    api_key=openai_key)

# MongoDBの設定
mongo_conn_str = os.getenv('COSMOS_CONNECTION_STRING') # MongoDBの接続文字列
url_parts = list(urlparse(mongo_conn_str)) # 接続文字列を分解
escaped_username = quote_plus(os.getenv('COSMOS_USERNAME')) # エスケープ：ユーザー名をURLエンコード
escaped_password = quote_plus(os.getenv('COSMOS_PASSWORD')) # エスケープ：パスワードをURLエンコード
url_parts[1] = url_parts[1].replace('<user>:<password>', f'{escaped_username}:{escaped_password}') # ユーザー名とパスワードを置換
mongo_conn_str = urlunparse(url_parts) # 接続文字列を再構築

# コマンドライン引数のパーサーを作成
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--index', type=str, help='Index for the query')
parser.add_argument('--text', type=str, help='Text input for the query')
# コマンドライン引数を解析
args = parser.parse_args()

mongo_db_name = 'db1'
mongo_collection_name = 'coll_holtest'
client = motor.motor_asyncio.AsyncIOMotorClient(mongo_conn_str)
db = client[mongo_db_name]
collection = db[mongo_collection_name]

embedding_model_name = 'embedding01'

async def main(): 
    # テキスト入力
    if args.text:
        text_input = args.text
    else:
        text_input = "アメリカ大統領選挙"
    # テキストをベクトルに変換
    vector = openai.embeddings.create(input=text_input,model=embedding_model_name).data[0].embedding
    
    # ベクトルを使用してMongoDBを検索
    # 集計ステージ : query1 .... ベクトル検索
    query1 = {
          '$search': {
            "cosmosSearch": {
                "vector": vector,
                "path": args.index,
                "k": 2,
              },
              "returnStoredSource": True 
          }
         }
    # 集計ステージ : query2 .... 表示項目のプロジェクション
    query2 = {
           '$project': { 
               "_id" : True,
               "name" : True,
               "num" : True,
               "SimScore": {
                  "$meta": "searchScore" 
               },
               "text" : True,
               "summary": True
           }
    }

    results = collection.aggregate(pipeline=[query1, query2])
    async for result in results:
        print("--- Similarity Score ({simscore}): {fname} ---".format(simscore=result["SimScore"], fname=result['name']))
        print("Summary:", result['summary'])

if __name__ == '__main__':
    asyncio.run(main())