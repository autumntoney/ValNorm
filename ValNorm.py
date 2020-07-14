#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 22:27:21 2020

@author: atoney
"""

import numpy as np
import pickle
import pandas as pd
import random 
from scipy.stats import norm
import math
from scipy.spatial.distance import cosine
import multiprocessing as mp


def findDeviation(array):
    array.sort()
    mean1 = np.mean(array)
    squareSum = 0
    
    for i in range(len(array)):
        squareSum += ((array[i] - mean1) ** 2)
    dev = math.sqrt((squareSum) / (len(array) - 1))
    return dev

def calculateCumulativeProbability(arr, value):
    cumulative=-100
    arr.sort()
    cumulative = norm.cdf(value, np.mean(arr), findDeviation(arr))
    return cumulative  
def getNullDistribution(idx, nullMatrix, setSize, iterations):
    distribution = [0]*iterations
    
    row = list(nullMatrix[idx])
    for itr in range(iterations):
        np.random.shuffle(row)
        break_length = int(setSize / 2)
        meanFirstAttribute = np.mean(row[0:break_length])
        meanSecondAttribute = np.mean(row[break_length:setSize])
        distribution[itr] = meanFirstAttribute - meanSecondAttribute
    return distribution

def getWordEmbedding(wordEmbeddingFile, word):
    return wordEmbeddingFile[word]
    
def effectSize(array, mean):
    d = findDeviation(array)
    es = mean / d     
    return es

def removeCategoryWordsIfNotInDictionary(vocab_array, semanticModel):
    corpus_words = list(semanticModel.keys())
    remove_words = []
    for wd in vocab_array:
        if wd not in corpus_words:
            print(wd)
            remove_words.append(wd)
    new_vocab_array = [x for x in vocab_array if x not in remove_words]
    return new_vocab_array

def cs_sim(x, y):
    return 1 - cosine(x, y)

VOCABULARY = None
BOTH_STEREOTYPES = None
TO_SHUFFLE = None
CONCEPT1_NULL_MATRIX = None
SEMANTIC_MODEL = None
MEAN_CONCEPT1_STEREOTYPE1 = None
MEAN_CONCEPT1_STEREOTYPE2 = None
ITERATIONS = None

def computeEffectSizeAndPVal(i):
    nullDistributionConcept1 = [0] * len(BOTH_STEREOTYPES)

    for itr in range(len(BOTH_STEREOTYPES)):
        col_val = TO_SHUFFLE[itr]
        item = CONCEPT1_NULL_MATRIX[i][col_val]
        
        nullDistributionConcept1[itr] = item
    
    nullDistribution = getNullDistribution(i, CONCEPT1_NULL_MATRIX,
                                           len(BOTH_STEREOTYPES), ITERATIONS) 
    
    delta = MEAN_CONCEPT1_STEREOTYPE1[i] - MEAN_CONCEPT1_STEREOTYPE2[i]
    e = effectSize(nullDistributionConcept1, delta)
    p = 1-calculateCumulativeProbability(nullDistribution, delta)
    print(i, "Vocabulary Word: ", VOCABULARY[i], "Effect Size: ", e, "P Val: ", p)    
    return (e, p)


def WordEmbeddingFactualAssociationTestVocab(semanticModel, vocabToTest):
    vocabulary = vocabToTest
    iterations = 10000
    wordDimension = 200
    
    pleasant = ["caress", "freedom", "health", "love", "peace", "cheer", "friend","heaven", 
                "loyal", "pleasure", "diamond", "gentle", "honest", "lucky", "rainbow","diploma",
                "gift", "honor", "miracle", "sunrise", "family", "happy", "laughter","paradise", 
                "vacation"]
    pleasant1 = removeCategoryWordsIfNotInDictionary(pleasant, semanticModel)

    unpleasant = ["abuse" , "crash" , "filth" , "murder" , "sickness" , "accident" , "death" , 
                  "grief" , "poison" , "stink" , "assault" , "disaster" , "hatred" , "pollute" , 
                  "tragedy" , "divorce" , "jail" , "poverty" , "ugly" , "cancer" , "kill" , 
                  "rotten" , "vomit" , "agony" , "prison"]

    unpleasant1 = removeCategoryWordsIfNotInDictionary(unpleasant, semanticModel)
    
    attributesFirstSet = pleasant1
    attributesSecondSet = unpleasant1
    
    if len(pleasant1) != len(unpleasant1):
        pleasant = random.shuffle(pleasant1)
        unpleasant = random.shuffle(unpleasant1)
        diff = abs(len(pleasant1) - len(unpleasant1))
        new_len = 25 - diff
        attributesFirstSet = pleasant1[0:new_len]
        attributesSecondSet = unpleasant1[0:new_len]
    
    
    #print(attributesFirstSet)
    #print(attributesSecondSet)
    
    vocabulary = removeCategoryWordsIfNotInDictionary(vocabulary, semanticModel)

    
    meanConcept1Stereotype1 = [0] * len(vocabulary)
    meanConcept1Stereotype2 = [0] * len(vocabulary)
    
    
    bothStereotypes = attributesFirstSet + attributesSecondSet
    random.shuffle(bothStereotypes) 
    
    toShuffle = [i for i in range(len(bothStereotypes))] 
    random.shuffle(toShuffle)
    
    random.shuffle(attributesFirstSet)
    
    random.shuffle(attributesSecondSet)
   

    #vocab to attributeFirstSet
    for i in range(len(vocabulary)):
        concept1Embedding = getWordEmbedding(semanticModel, vocabulary[i])
        
        for j in range(len(attributesFirstSet)):
            stereotype1Embedding = getWordEmbedding(semanticModel, attributesFirstSet[j])
            similarityCompatible = cs_sim(concept1Embedding, stereotype1Embedding)
            meanConcept1Stereotype1[i] += similarityCompatible
        meanConcept1Stereotype1[i] /= (len(attributesFirstSet))
    #print(meanConcept1Stereotype1)
       
    #vocab to attributeSecondSet
    for i in range(len(vocabulary)):
        concept1Embedding = getWordEmbedding(semanticModel, vocabulary[i])
        
        for j in range(len(attributesSecondSet)):
            stereotype2Embedding = getWordEmbedding(semanticModel, attributesSecondSet[j])
            similarityCompatible = cs_sim(concept1Embedding, stereotype2Embedding)
            meanConcept1Stereotype2[i] += similarityCompatible
        meanConcept1Stereotype2[i] /= (len(attributesSecondSet))
    
    #print(meanConcept1Stereotype2)

    
    concept1NullMatrix = np.zeros((len(vocabulary), len(bothStereotypes)))
    
      
    
    for i in range(len(vocabulary)):
        concept1Embedding = getWordEmbedding(semanticModel, vocabulary[i])
        for j in range(len(bothStereotypes)):
            nullEmbedding = getWordEmbedding(semanticModel, bothStereotypes[j])
            similarityCompatible = cs_sim(concept1Embedding, nullEmbedding)
            concept1NullMatrix[i][j]=similarityCompatible
       
    global VOCABULARY
    VOCABULARY = vocabulary
    global BOTH_STEREOTYPES
    BOTH_STEREOTYPES = bothStereotypes
    global TO_SHUFFLE
    TO_SHUFFLE = toShuffle
    global CONCEPT1_NULL_MATRIX
    CONCEPT1_NULL_MATRIX = concept1NullMatrix
    global SEMANTIC_MODEL
    SEMANTIC_MODEL = semanticModel
    global MEAN_CONCEPT1_STEREOTYPE1
    MEAN_CONCEPT1_STEREOTYPE1 = meanConcept1Stereotype1
    global MEAN_CONCEPT1_STEREOTYPE2
    MEAN_CONCEPT1_STEREOTYPE2 = meanConcept1Stereotype2
    global ITERATIONS
    ITERATIONS = iterations
    
    print("starting multiprocessing")

    with mp.Pool(processes=8) as pool:
        items = list(range(len(vocabulary)))
        
        print("running pool.map")
        output = pool.map(computeEffectSizeAndPVal, items)
        
        effect_size_vector = list(map(lambda x: x[0], output))
        pval_vector = list(map(lambda x: x[1], output))
       
        WEFAT_Results = pd.DataFrame()
        WEFAT_Results["word"] = vocabulary
        WEFAT_Results["effect_size"] = list(effect_size_vector)
        WEFAT_Results["p_value"] = list(pval_vector)
        
        return WEFAT_Results
