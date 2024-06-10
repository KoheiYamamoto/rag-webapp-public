import streamlit as st, os, motor.motor_asyncio, asyncio
from urllib.parse import quote_plus, urlparse, urlunparse
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()

### 任意選択
index = "summary_vectors" # "summary_vectors" | "vectors"

st.title("Rag Chat") # タイトルの設定

### INSTANCE
USER_NAME = "user"
ASSISTANT_NAME = "assistant"
client_vectorise = AzureOpenAI(
            azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT_VECTORISE'), 
            api_key = os.getenv('AZURE_OPENAI_API_KEY_VECTORISE'),  
            api_version = "2023-12-01-preview", 
        )
client_summarise = AzureOpenAI(
            azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT_SUMMARISE'), 
            api_key = os.getenv('AZURE_OPENAI_API_KEY_SUMMARISE'),  
            api_version = "2023-12-01-preview", 
        )
# Mongo DB の設定
mongo_conn_str = os.getenv('COSMOS_CONNECTION_STRING') # MongoDBの接続文字列
url_parts = list(urlparse(mongo_conn_str)) # 接続文字列を分解
escaped_username = quote_plus(os.getenv('COSMOS_USERNAME')) # エスケープ：ユーザー名をURLエンコード
escaped_password = quote_plus(os.getenv('COSMOS_PASSWORD')) # エスケープ：パスワードをURLエンコード
url_parts[1] = url_parts[1].replace('<user>:<password>', f'{escaped_username}:{escaped_password}') # ユーザー名とパスワードを置換
mongo_conn_str = urlunparse(url_parts) # 接続文字列を再構築
mongo_db_name = 'db1'
mongo_collection_name = 'coll_holtest'
client = motor.motor_asyncio.AsyncIOMotorClient(mongo_conn_str)
db = client[mongo_db_name]
collection = db[mongo_collection_name]

if "openai_model" not in st.session_state: # セッション内で使用するモデルが指定されていない場合のデフォルト値
    st.session_state["openai_model"] = os.getenv('AZURE_OPENAI_DEPLOYMENT_SUMMARISE')
if "messages" not in st.session_state: # セッション内のメッセージが指定されていない場合のデフォルト値
    st.session_state.messages = []
    st.session_state.prompt_completions = []
if "Clear" not in st.session_state: # セッション内でチャット履歴をクリアするかどうかの状態変数
    st.session_state.Clear = False
for message in st.session_state.messages: # 以前のメッセージを表示
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

embedding_model_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_VECTORISE') # OpenAI Studioでデプロイしたモデルの名前

# 検索結果を含んだプロンプトの作成
async def rag_prompt(user_input):
    vector = client_vectorise.embeddings.create(input=user_input,model=embedding_model_name).data[0].embedding
    
    # ベクトルを使用してMongoDBを検索
    # 集計ステージ : query1 .... ベクトル検索
    query1 = {
          '$search': {
            "cosmosSearch": {
                "vector": vector,
                "path": index,
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
    search_result = ""
    async for result in results:
        search_result += "--- Similarity Score ({simscore}): {fname} ---".format(simscore=result["SimScore"], fname=result['name']) + "\n"
        search_result += "Summary: {}".format(result['summary']) + "\n"
    print(search_result)
    
    prompt_user = """
        以下の内容について回答してください。
        - user input : {user_input} 
        - search results : {search_result}
        """.format(user_input=user_input, search_result=search_result)
    return {"role": "user", "content": prompt_user}

prompt_system = """
                あなたは、優秀なチャットアシスタントです。以下の指示に従ってください。
                指示 1: あなたはユーザとの会話に友達のように答えます。
                指示 2: ユーザからの質問には、その質問の回答に関連するヒント (search_result) が付帯します。回答の中で関連性が"極めて"高いと判断した場合でのみ、ヒントを使用してください。
                指示 3: ただし、関連性が肌感で90%以下のヒントであれば、絶対に無視してください。その場合は、ユーザと友達のように普段の対話をしてください。検索結果が見つからなったなどの不要な言葉は不要です。
                
                # Example 1
                - user input : アメリカの大統領選挙にでたいなぁ
                - search results : --- Similarity Score (0.8996990143714717): ---
                                    Summary: アメリカ合衆国大統領は、国家元首であり行政府の長である。選挙資格は、35歳以上でアメリカ合衆国国内における在留期間が14年以上で、出生によるアメリカ合衆国市民権保持者であることが必要である。大統領選挙 は、4年に1度実施され、国民投票によって選出される。大統領には、連邦議会を通過した法案への拒否権や法律制定に関する勧告権を持っており、軍の最高指揮権も保持している。また、大統領には行政権が帰属しているが、権力分立を徹底し、権力相互間の抑制・均衡が働いている。大統領は、最長で連続・返り咲きを問わず2期、8年まで務めることができる。なお、大統領選挙は、間接選挙であり、選挙人団によって大統領及び副大統領がペアで選出されるが、一般有権者は直接選挙に近い形で投票することになる。しかし、過半数の選挙人を獲得できない場合には、連邦議会の下院及び上院がそれぞれ大統領及び副大統領を選出することもあり得る。
                                    --- Similarity Score (0.8701136708259583): ---
                                    Summary: アメリカ合衆国大統領の継承順位は、1947年に制定された「大統領継承法」によって設定されており、副大統領、上院議長、内閣の閣僚らが定められている。しかし、移民から大統領になる場合には規定の資格を満たさ ないため、順位内にいる場合は飛ばされる。歴代大統領は様々な出身地や人種、経歴を持ち、弁護士出身者が最も多い。大統領の呼称は「ミスター・プレジデント」と略称「サー」で、女性大統領の場合には「マダム・プレジデント」と略称「マァム」となる。存命の前・元大統領全員が同様に「ミスター・プレジデント」として接遇される。
                - your answer : いいね!! ちなみに、選挙資格は、アメリカ合衆国国内における在留期間が14年以上で、出生によるアメリカ合衆国市民権保持者であることが必要だよ!! (1000.txt)
                # Example 2
                - user input : おはよう！
                - search results : --- Similarity Score (0.8996990143714717): ---
                                    Summary: アメリカ合衆国大統領は、国家元首であり行政府の長である。選挙資格は、35歳以上でアメリカ合衆国国内における在留期間が14年以上で、出生によるアメリカ合衆国市民権保持者であることが必要である。大統領選挙 は、4年に1度実施され、国民投票によって選出される。大統領には、連邦議会を通過した法案への拒否権や法律制定に関する勧告権を持っており、軍の最高指揮権も保持している。また、大統領には行政権が帰属しているが、権力分立を徹底し、権力相互間の抑制・均衡が働いている。大統領は、最長で連続・返り咲きを問わず2期、8年まで務めることができる。なお、大統領選挙は、間接選挙であり、選挙人団によって大統領及び副大統領がペアで選出されるが、一般有権者は直接選挙に近い形で投票することになる。しかし、過半数の選挙人を獲得できない場合には、連邦議会の下院及び上院がそれぞれ大統領及び副大統領を選出することもあり得る。
                                    --- Similarity Score (0.8701136708259583): ---
                                    Summary: アメリカ合衆国大統領の継承順位は、1947年に制定された「大統領継承法」によって設定されており、副大統領、上院議長、内閣の閣僚らが定められている。しかし、移民から大統領になる場合には規定の資格を満たさ ないため、順位内にいる場合は飛ばされる。歴代大統領は様々な出身地や人種、経歴を持ち、弁護士出身者が最も多い。大統領の呼称は「ミスター・プレジデント」と略称「サー」で、女性大統領の場合には「マダム・プレジデント」と略称「マァム」となる。存命の前・元大統領全員が同様に「ミスター・プレジデント」として接遇される。
                - your answer : おはようー！今日はいい天気だね！
                # Example 3
                - user input : おはよう！
                - search results : ---- Similarity Score (0.7718947393831579):  ---
                                    Summary: 柿内ゆきちは日本の漫画家であり、別名“ゆきち先生”でも知られている。彼の作品は多くの読者に支持され、多くの漫画ファンから愛されている。
                                    --- Similarity Score (0.7687552113082893):  ---
                                    Summary: このパッセージは、第二次世界大戦中に起こった、アメリカ海軍の「ポートシカゴ爆発事件」に関する情報源のリストを提供しています。関連書籍やドキュメンタリー映画、テレビドラマの情報が含まれており、また、事件に関する学習用リソースも紹介されています。事件の詳細や背景に関する情報は含まれていま せん。
                - your answer : こんにちはー！調子はどう？
                """

# ユーザーからの新しい入力を取得
if prompt := st.chat_input("何かお困りですか?"):
    with st.chat_message(USER_NAME):
        st.markdown(prompt)

    # ユーザの入力内容とユーザプロンプト（検索結果を含む）を作成
    with st.spinner('アシスタントが考え中...'):
        prompt_with_search = asyncio.run(rag_prompt(prompt))
    # prompt_with_search = asyncio.run(rag_prompt(prompt))

    # セッション保持用のリストに入力内容を追加
    # セッション保持用のリストにプロンプト（ユーザには見えない）を追加
    st.session_state.messages.append({"role": USER_NAME, "content": prompt})
    st.session_state.prompt_completions.append(prompt_with_search)

    with st.chat_message(ASSISTANT_NAME):
        message_placeholder = st.empty() # 一時的なプレースホルダーを作成
        full_response = ""
        # ChatGPTからのストリーミング応答を処理
        for response in client_summarise.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": "system", "content": prompt_system},
                *[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.prompt_completions
                ],
            ],
            temperature=0,
            stream=True,
        ):
            # Check if 'response.choices' is not empty before accessing its elements
            if response.choices and response.choices[0].delta.content is not None:
                full_response += response.choices[0].delta.content.replace('\n', ' ')
                message_placeholder.markdown(full_response + "▌") # Display the intermediate result of the response
                message_placeholder.markdown(full_response) # Display the final response
    
    # セッション保持用のリストに回答を追加
    st.session_state.messages.append({"role": ASSISTANT_NAME, "content": full_response})
    st.session_state.prompt_completions.append({"role": ASSISTANT_NAME, "content": full_response})

    st.session_state.Clear = True # チャット履歴のクリアボタンを有効にする

# チャット履歴をクリアするボタンが押されたら、メッセージをリセット
if st.session_state.Clear:
    if st.button('会話履歴を削除する'):
        
        # セッション保持用のリストを両方ともリセット
        st.session_state.messages = [] # メッセージのリセット
        st.session_state.prompt_completions = [] # チャット履歴のリセット

        full_response = ""
        st.session_state.Clear = False # クリア状態をリセット
        st.experimental_rerun() # 画面を更新