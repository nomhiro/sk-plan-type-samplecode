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
from semantic_kernel.planning.sequential_planner import SequentialPlanner
import asyncio

async def main():
    
    kernel = sk.Kernel()

    # 環境変数からOpenAIの設定を取得
    deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()

    kernel.add_chat_service(
        "chat_completion",
        AzureChatCompletion(deployment_name=deployment, api_key=api_key, endpoint=endpoint)
    )
    
    ask = """以下の文章にタイトルをつけて。タイトルはフランス語にしてほしい。

AIモデルは、ユーザー向けのメッセージや画像を簡単に生成できます。これは、シンプルなチャットアプリを構築する際には役立ちますが、ビジネスプロセスを自動化し、ユーザーがより多くのことを達成できるようにする、完全に自動化されたAIエージェントを構築するだけでは十分ではありません。そのためには、これらのモデルから応答を受け取り、それらを使用して既存のコードを呼び出し、実際に生産的なことを実行できるフレームワークが必要です。
セマンティックカーネルでは、まさにそれを実現しました。既存のコードを AI モデルに簡単に記述して、AI モデルが呼び出しを要求できるようにする SDK を作成しました。その後、セマンティック カーネルは、モデルの応答をコードの呼び出しに変換するという面倒な作業を行います。"""

    #ask = "98と812を足して、2で割ってください。"

    # プラグインの読み込み
    plugins_directory = "./plugins/"
    summarize_plugin = kernel.import_semantic_plugin_from_directory(plugins_directory, "SummarizePlugin")
    writer_plugin = kernel.import_semantic_plugin_from_directory(plugins_directory, "WriterPlugin")

    # ActionPlannerのインスタンスを生成
    planner = SequentialPlanner(kernel)
    # プランの生成
    sequential_plan = await planner.create_plan(goal=ask)
    print("### generated plan ###")
    for step in sequential_plan._steps:
        print("■ step: ", sequential_plan._steps.index(step) + 1, "/", len(sequential_plan._steps))
        print(step.description, ": ", step._state.__dict__)
        print("input: ", step._parameters.input)
    
    # プランの実行
    results = await sequential_plan.invoke()
    print("\n\n### results ###")
    print(results)

asyncio.run(main())
