import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
)
from semantic_kernel.core_plugins.text_plugin import TextPlugin
from semantic_kernel.planning.basic_planner import BasicPlanner
import asyncio

async def main():
    
    kernel = sk.Kernel()

    # 環境変数からOpenAIの設定を取得
    deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()

    kernel.add_chat_service(
        "chat_completion",
        AzureChatCompletion(deployment_name=deployment, api_key=api_key, endpoint=endpoint)
    )
    
    ask = """以下の文章にタイトルをつけて。

AIモデルは、ユーザー向けのメッセージや画像を簡単に生成できます。これは、シンプルなチャットアプリを構築する際には役立ちますが、ビジネスプロセスを自動化し、ユーザーがより多くのことを達成できるようにする、完全に自動化されたAIエージェントを構築するだけでは十分ではありません。そのためには、これらのモデルから応答を受け取り、それらを使用して既存のコードを呼び出し、実際に生産的なことを実行できるフレームワークが必要です。
セマンティックカーネルでは、まさにそれを実現しました。既存のコードを AI モデルに簡単に記述して、AI モデルが呼び出しを要求できるようにする SDK を作成しました。その後、セマンティック カーネルは、モデルの応答をコードの呼び出しに変換するという面倒な作業を行います。"""

    # プラグインの読み込み
    plugins_directory = "./plugins/"
    summarize_plugin = kernel.import_semantic_plugin_from_directory(plugins_directory, "SummarizePlugin")
    text_plugin = kernel.import_plugin(TextPlugin(), "TextPlugin")

    planner = BasicPlanner()
    # プランの生成
    basic_plan = await planner.create_plan(ask, kernel)
    print("### generated plan ###")
    print(basic_plan.generated_plan)
    
    # プランの実行
    results = await planner.execute_plan(basic_plan, kernel)
    print("\n\n### results ###")
    print(results)

asyncio.run(main())
