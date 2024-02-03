import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
)
from semantic_kernel.core_plugins import (
    FileIOPlugin,
    MathPlugin,
    TextPlugin,
    TimePlugin,
)
from semantic_kernel.planning.stepwise_planner import StepwisePlanner
from semantic_kernel.planning.stepwise_planner.stepwise_planner_config import (
    StepwisePlannerConfig,
)
from semantic_kernel.connectors.search_engine import BingConnector
import asyncio
from plugins.WebSearchEnginePlugin.WebSearchEnginePlugin import WebSearchEnginePlugin

async def main():
    
    kernel = sk.Kernel()

    # 環境変数からOpenAIの設定を取得
    deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()

    kernel.add_chat_service(
        "chat_completion",
        AzureChatCompletion(deployment_name=deployment, api_key=api_key, endpoint=endpoint)
    )
    
    # プラグインの読み込み
    plugins_directory = "./plugins/"
    BING_API_KEY = sk.bing_search_settings_from_dot_env()
    connector = BingConnector(BING_API_KEY)
    kernel.import_plugin(WebSearchEnginePlugin(connector), plugin_name="WebSearch")
    kernel.import_plugin(TimePlugin(), "time")
    kernel.import_plugin(MathPlugin(), "math")
    kernel.import_semantic_plugin_from_directory(plugins_directory, "SummarizePlugin")
    kernel.import_semantic_plugin_from_directory(plugins_directory, "WriterPlugin")
    
    #ask = """can you recommend a good digital camera for $1000 ? From Wikipedia summarise an article about the manufacturer."""
#     ask = """以下の歌詞の音楽に英語のタイトルをつけて。

# 沈むように溶けてゆくように
# 二人だけの空が広がる夜に
# 「さよなら」だけだった
# その一言で全てが分かった
# 日が沈み出した空と君の姿
# フェンス越しに重なっていた
# 初めて会った日から
# 僕の心の全てを奪った
# どこか儚い空気を纏う君は
# 寂しい目をしてたんだ
# いつだってチックタックと
# 鳴る世界で何度だってさ
# 触れる心無い言葉うるさい声に
# 涙が零れそうでも
# ありきたりな喜び
# きっと二人なら見つけられる
# 騒がしい日々に笑えない君に
# 思い付く限り眩しい明日を
# 明けない夜に落ちてゆく前に
# 僕の手を掴んでほら
# 忘れてしまいたくて閉じ込めた日々も
# 抱きしめた温もりで溶かすから
# 怖くないよいつか日が昇るまで
# 二人でいよう
# 君にしか見えない
# 何かを見つめる君が嫌いだ
# 見惚れているかのような恋するような
# そんな顔が嫌いだ
# 信じていたいけど信じれないこと
# そんなのどうしたってきっと
# これからだっていくつもあって
# そのたんび怒って泣いていくの
# それでもきっといつかはきっと僕らはきっと
# 分かり合えるさ信じてるよ
# もう嫌だって疲れたんだって
# がむしゃらに差し伸べた僕の手を振り払う君
# もう嫌だって疲れたよなんて
# 本当は僕も言いたいんだ
# ほらまたチックタックと
# 鳴る世界で何度だってさ
# 君の為に用意した言葉どれも届かない
# 「終わりにしたい」だなんてさ
# 釣られて言葉にした時
# 君は初めて笑った
# 騒がしい日々に笑えなくなっていた
# 僕の目に映る君は綺麗だ
# 明けない夜に溢れた涙も
# 君の笑顔に溶けていく
# 変わらない日々に泣いていた僕を
# 君は優しく終わりへと誘う
# 沈むように溶けてゆくように
# 染み付いた霧が晴れる
# 忘れてしまいたくて閉じ込めた日々に
# 差し伸べてくれた君の手を取る
# 涼しい風が空を泳ぐように今吹き抜けていく
# 繋いだ手を離さないでよ
# 二人今、夜に駆け出していく"""

    #ask = """12と918を足して、3で割ってから4掛けてください。"""
    
    ask = """以下の文章にタイトルをつけて。タイトルはフランス語にしてほしい。

AIモデルは、ユーザー向けのメッセージや画像を簡単に生成できます。これは、シンプルなチャットアプリを構築する際には役立ちますが、ビジネスプロセスを自動化し、ユーザーがより多くのことを達成できるようにする、完全に自動化されたAIエージェントを構築するだけでは十分ではありません。そのためには、これらのモデルから応答を受け取り、それらを使用して既存のコードを呼び出し、実際に生産的なことを実行できるフレームワークが必要です。
セマンティックカーネルでは、まさにそれを実現しました。既存のコードを AI モデルに簡単に記述して、AI モデルが呼び出しを要求できるようにする SDK を作成しました。その後、セマンティック カーネルは、モデルの応答をコードの呼び出しに変換するという面倒な作業を行います。"""

    # ActionPlannerのインスタンスを生成
    planner = StepwisePlanner(kernel, StepwisePlannerConfig(max_iterations=10, min_iteration_time_ms=1000))
    # プランの生成
    stepwise_plan = planner.create_plan(goal=ask)
    
    print("### generated plan ###")
    for index, step in enumerate(stepwise_plan._steps):
        print("Step:", index)
        print("Description:", step.description)
        print("Function:", step.plugin_name + "." + step._function.name)
    
    # プランの実行
    results = await stepwise_plan.invoke()
    print("\n\n### results ###")
    print(results)
    
    print("\n\n### see steps ###")
    for index, step in enumerate(stepwise_plan._steps):
        print("Step:", index)
        print("Description:", step.description)
        print("Function:", step.plugin_name + "." + step._function.name)
        if len(step._outputs) > 0:
            print("  Output:\n", str.replace(results[step._outputs[0]], "\n", "\n  "))

asyncio.run(main())
