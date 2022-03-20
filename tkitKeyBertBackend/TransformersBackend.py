from keybert.backend import BaseEmbedder
# from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
import torch
# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    """
    由于bert的cls作为短语的表示效果不是很理想，而且后期非官方版本的都取消列下一句任务，所所以这里采用均值池化策略作为短语表示。
    pooling策略
    SBERT在BERT/RoBERTa的输出结果上增加了一个pooling操作，从而生成一个固定大小的句子embedding向量。实验中采取了三种pooling策略做对比：

    直接采用CLS位置的输出向量代表整个句子的向量表示
    MEAN策略，计算各个token输出向量的平均值代表句子向量
    MAX策略，取所有输出向量各个维度的最大值代表句子向量




    参考论文：
    https://arxiv.org/abs/1908.10084
    中文注释地址
    https://www.cnblogs.com/gczr/p/12874409.html

    """
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class TransformersBackend(BaseEmbedder):
    def __init__(self, embedding_model,tokenizer):
        super().__init__()
        self.embedding_model = embedding_model
        self.tokenizer=tokenizer

    def embed(self, documents, verbose=False):
#         embeddings = self.embedding_model.encode(documents, show_progress_bar=verbose)
# text = "用你喜欢的任何文本替换我。"
        maxLen=12
        encoded_input = self.tokenizer(documents,padding="max_length", max_length=maxLen, truncation=True,return_tensors="pt")
        output = model(**encoded_input)

#         print(output.pooler_output.cpu().detach().numpy().shape)
#         return output.pooler_output.cpu().detach().numpy()
        return mean_pooling(output, encoded_input["attention_mask"]).cpu().detach().numpy()

if __name__ == '__main__':
    """
    示例
    https://www.kaggle.com/terrychanorg/keybert-extract-keywords-notebookcb54da42f2
    """
    from keybert import KeyBERT
    from transformers import BertTokenizer, BertModel
    import jieba
    # from tkitKeyBertBackend.TransformersBackend import TransformersBackend

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
    # [('功能', 0.9146), ('提供', 0.8984), ('需要', 0.86), ('使用', 0.8554), ('没有', 0.8519)]