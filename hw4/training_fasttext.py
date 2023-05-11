from gensim.models import FastText
import multiprocessing
max_cpu_counts = multiprocessing.cpu_count()
word_dim_size = 300 
WIKI_SEG_TXT = "wiki_seg.txt"
#讀取訓練語句
# sentences = word2vec.LineSentence(WIKI_SEG_TXT)

# 訓練模型
model = FastText(vector_size=word_dim_size, workers=max_cpu_counts,min_count=1)
model.build_vocab(corpus_iterable=WIKI_SEG_TXT)
model.train(corpus_iterable=WIKI_SEG_TXT, total_examples=len(WIKI_SEG_TXT), epochs=10)  # train
# 儲存模型
output_model = f"FastText.zh.{word_dim_size}.model"
model.save(output_model)