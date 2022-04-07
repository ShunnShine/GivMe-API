import gluonnlp as nlp
import numpy as np
import os
from sklearn.cluster import KMeans

#Fill the embedding dictionary.
glove_embedding = {}
for i in range(4):
     with open(f"Glove300dEmbeddings/words{1+i*25000}_{(i+1)*25000}.txt", "r", encoding='utf-8') as read:
          for line in read:
               arr = line.split()
               glove_embedding[arr[0]] = np.array(list(map(lambda x: float(x), arr[1:])))

# Built with help from: 
# Principal Component Analysis with NumPy
# An article written by Wendy Navarrete
# https://towardsdatascience.com/pca-with-numpy-58917c1d0391


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


def import_word_list(word_list):
     word_list_embedding = []
     cant_find = []
     found = []
     for word in word_list:
          word = word.lower()
          if word in glove_embedding:
               word_list_embedding.append(glove_embedding[word])
               found.append(word)
          else:
               cant_find.append(word)
     return found, cant_find, np.array(word_list_embedding, np.float32)

def k_means_clus(wordEmbedding, num_of_clus):
     return KMeans(n_clusters=num_of_clus).fit(wordEmbedding).labels_
               

def PCA(list_of_words):

     found_words_vocab, cant_find_words, wordlist_Embedding = import_word_list(list_of_words)
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
     list1_found_words_vocab, cant_find_words, list1_Embedding = import_word_list(word_list1)
     list2_found_words_vocab, cant_find_words_temp, list2_Embedding = import_word_list(word_list2)

     cant_find_words += cant_find_words_temp

     list1_mean = np.mean(list1_Embedding, axis=0)
     list2_mean = np.mean(list2_Embedding, axis = 0)

     mean_direction = list1_mean - list2_mean
     mean_unit_vector = mean_direction / np.linalg.norm(mean_direction)
     x_values = []
     for i in range(len(list1_Embedding)):
          dot_product = np.dot(list1_Embedding[i], mean_unit_vector)
          x_values.append(float(dot_product))
          list1_Embedding[i] = list1_Embedding[i] - mean_unit_vector * dot_product

     for i in range(len(list2_Embedding)):
          dot_product = np.dot(list2_Embedding[i], mean_unit_vector)
          x_values.append(float(dot_product))
          list2_Embedding[i] = list2_Embedding[i] - mean_unit_vector * dot_product
     
     both_list_embedding = np.vstack((list1_Embedding, list2_Embedding))

     both_list_embedding = standardize_data(both_list_embedding)

     # compute eigen matrix
     covariance_matrix = np.cov(both_list_embedding.T)
     eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)


     # # Calculating the explained variance on each of components
     # variance_explained = []
     # for i in eigen_values:
     #      variance_explained.append((i/sum(eigen_values))*100)
        
     # cumulative_variance_explained = np.cumsum(variance_explained)

     projection_matrix = (eigen_vectors.T[:][:2]).T
     projected_words = both_list_embedding.dot(projection_matrix)

     variance_explained = []
     for i in eigen_values:
          variance_explained.append((i/sum(eigen_values)).real*100)

     #final step: get the projected coords
     return list1_found_words_vocab, list2_found_words_vocab, cant_find_words, projected_words.real, x_values, sum(variance_explained[:3]), sum(variance_explained[:2])

          

if __name__ == '__main__':
     Two_Means(["man", "male", "he", "him"], ["she", "her", "female"])
