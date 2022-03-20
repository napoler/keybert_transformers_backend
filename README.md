# keybert_transformers_backend
keybert_transformers_backend  

由于默认只支持SentenceTransformer模型导入，无法直接使用transformers作为后端，还好官方支持自定义方案导入。这里提供keybert引入huggingface transformers作为后端，可以方便处理中文

https://github.com/napoler/keybert_transformers_backend

```python

"""
    示例
    https://www.kaggle.com/terrychanorg/keybert-extract-keywords-notebookcb54da42f2
    """
from keybert import KeyBERT
import jieba
from tkitKeyBertBackend.TransformersBackend import TransformersBackend
from transformers import BertTokenizer, BertModel
doc = """
    1.没有提供分词功能，英文是空格分词，中文输入需要分完词输入。
    2.选择候选词：默认使用CountVectorizer进行候选词选择。
    3.  model：默认方式，候选词向量和句向量的距离排序。
        mmr：最大 边际距离 方法，保证关键词之间的多样性。考虑词之间的相似性。
        max_sum：候选词之间相似和最小的组合。

          """
seg_list = jieba.cut(doc, cut_all=True)
doc = " ".join(seg_list)
# kw_model = KeyBERT()
# keywords = kw_model.extract_keywords(doc)

tokenizer = BertTokenizer.from_pretrained('uer/chinese_roberta_L-2_H-128')
model = BertModel.from_pretrained("uer/chinese_roberta_L-2_H-128")

custom_embedder = TransformersBackend(embedding_model=model,tokenizer=tokenizer)
# Pass custom backend to keybert
kw_model = KeyBERT(model=custom_embedder)
print(kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 1), stop_words=None))
```
	[('功能', 0.9146), ('提供', 0.8984), ('需要', 0.86), ('使用', 0.8554), ('没有', 0.8519)]


https://github.com/MaartenGr/KeyBERT
