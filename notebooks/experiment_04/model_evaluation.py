# 예시: 평가 결과 표
from IPython.display import display, Markdown

display(Markdown("""
|Epoch|Training Loss|Validation Loss|Wer|Cer|
|--|--|--|--|--|
|1|0.610700|0.195249|0.426034|0.154628|
|2|0.109000|-0.042550|0.259833|0.091110|
|3|-0.060200|-0.146214|0.206438|0.074300|
|4|-0.151900|-0.224590|0.190126|0.075191|
"""))