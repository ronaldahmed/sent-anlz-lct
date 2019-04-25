import nltk
import html
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pdb



class DataLoader:
  def __init__(self):
    np.random.seed(42)
    self.num_test_unlb = 5000
    self.num_dev = 5000

    print("Loading movie-training")
    train_chunk = self.load_csv("amazon_balanced_two_domains/movie_reviews-train.csv")
    self.train,self.dev_src = train_test_split(train_chunk,test_size=0.1)

    print("Loading movie-test")
    self.test_src  = self.load_csv("amazon_balanced_two_domains/movie_reviews-test.csv")
    print("Loading video-games data")
    test_chunk  = self.load_csv("amazon_balanced_two_domains/video_games-test.csv")
    self.tgt_unlb, self.dev_tgt, self.test_tgt = \
          self.split_tgt_data(test_chunk)


  def load_csv(self,filename):
    data_df = pd.read_csv(filename)
    data = []
    count = 1
    for pol,summ,revw in zip(data_df["polarity"],data_df["summary"],data_df["reviewText"]):
      data.append([pol,summ,revw])
      if count % 5000 == 0:
        print("-->",count)
      count += 1
    #
    return data

  def split_tgt_data(self,data):
    data = np.array(data)
    np.random.shuffle(data)
    unlb = data[:self.num_test_unlb]
    dev = data[self.num_test_unlb:(self.num_dev + self.num_test_unlb)]
    test = data[(self.num_dev + self.num_test_unlb):]
    return unlb,dev,test


  def dump_sopa_format(self,):
    for split,data in zip(["train","dev_src","dev_tgt","test_src","test_tgt","test_unlb"], \
                          [self.train,self.dev_src,self.dev_tgt,self.test_src, \
                          self.test_tgt,self.tgt_unlb]):
      print("Dumping %s..." % (split) )
      outdata = "soft_patterns/%s.data" % split
      outlab = "soft_patterns/%s.labels" % split
      outfile_data = open(outdata,'w')
      outfile_label = open(outlab,'w')
      for pol,_,revw in data:
        if pd.isnull(revw):
          continue
        label = 1 if pol=="positive" else 0
        revw = nltk.word_tokenize(html.unescape(revw.lower()))
        
        print(" ".join(revw),file=outfile_data)
        print(label,file=outfile_label)
      outfile_data.close()
      outfile_label.close()
    #

