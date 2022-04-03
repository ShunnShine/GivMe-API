import mxnet as mx
import gluonnlp as nlp
import numpy as np
import os
from sklearn.cluster import KMeans

# Built with help from: 
# Principal Component Analysis with NumPy
# An article written by Wendy Navarrete
# https://towardsdatascience.com/pca-with-numpy-58917c1d0391

###########################
# Load/Read Word Embedding
###########################
def loadDataVocab(embedType, preTrainFile):

     fasttext_2M300d = nlp.embedding.create(embedType, source=preTrainFile)

     # create vocabulary by using all the tokens
     vocab = nlp.Vocab(nlp.data.Counter(fasttext_2M300d.idx_to_token))
     vocab.set_embedding(fasttext_2M300d)
     #len(vocab.idx_to_token)
     count_tok = nlp.data.Counter(fasttext_2M300d.idx_to_token)
     wordsVocab = [x[0] for x in count_tok.most_common()]

     return vocab, wordsVocab[1:]

def standardize_data(arr):
         
    '''
    This function standardize an array, its substracts mean value, 
    and then divide the standard deviation.
    
    param 1: array 
    return: standardized array
    '''    
    rows, columns = arr.shape
    
    standardizedArray = np.zeros(shape=(rows, columns))
    tempArray = np.zeros(rows)
    
    for column in range(columns):
        
        mean = np.mean(arr[:,column])
        std = np.std(arr[:,column])
        tempArray = np.empty(0)
        
        for element in arr[:,column]:
            
            tempArray = np.append(tempArray, ((element - mean) / std))
 
        standardizedArray[:,column] = tempArray
    
    return standardizedArray


def import_word_list(word_list, wordVocab, wordEmbedding):
     word_list_embedding = []
     cant_find = []
     found = []
     for word in word_list:
          word = word.lower()
          try:
               i = wordVocab.index(word)
               word_list_embedding.append(list(wordEmbedding[i]))
               found.append(word)
          except ValueError:
               cant_find.append(word)
     return found, cant_find, np.array(word_list_embedding, np.float32)

def k_means_clus(wordEmbedding, num_of_clus):
     return KMeans(n_clusters=num_of_clus).fit(wordEmbedding).labels_
               

def PCA(list_of_words):
     embedType, preTrainFile = "glove", "glove.6B.300d"
     vocab, all_wordsVocab = loadDataVocab(embedType, preTrainFile)

     all_wordsVocab  = all_wordsVocab[:100000]
     all_WordEmbedding = vocab.embedding[all_wordsVocab].asnumpy()

     found_words_vocab, cant_find_words, wordlist_Embedding = import_word_list(list_of_words, all_wordsVocab, all_WordEmbedding)
     # standardized it
     wordlist_Embedding = standardize_data(wordlist_Embedding)

     # compute eigen matrix
     covariance_matrix = np.cov(wordlist_Embedding.T)
     eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)


     # # Calculating the explained variance on each of components
     # variance_explained = []
     # for i in eigen_values:
     #      variance_explained.append((i/sum(eigen_values))*100)
        
     # cumulative_variance_explained = np.cumsum(variance_explained)

     projection_matrix = (eigen_vectors.T[:][:3]).T
     projected_words = wordlist_Embedding.dot(projection_matrix)

     variance_explained = []
     for i in eigen_values:
          variance_explained.append((i/sum(eigen_values)).real*100)

     #final step: get the projected coords
     return found_words_vocab, cant_find_words, projected_words.real, sum(variance_explained[:3]), sum(variance_explained[:2])

def Two_Means(word_list1, word_list2):
     embedType, preTrainFile = "glove", "glove.6B.300d"
     vocab, all_wordsVocab = loadDataVocab(embedType, preTrainFile)

     embedding_Vocab  = all_wordsVocab[:100000]
     embedding_vectors = vocab.embedding[all_wordsVocab].asnumpy()

     found_words_vocab, cant_find_words, wordlist_Embedding = import_word_list(word_list1, all_wordsVocab, )