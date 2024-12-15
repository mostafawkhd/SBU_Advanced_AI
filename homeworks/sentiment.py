import random
import collections
import math
import sys
from util import *

def getWordFeatures(inputString):
    """
    Extracts the word features from a given string.
    Words are identified by whitespace delimiters.

    @param inputString: The input string to process.
    @return: A dictionary where keys are words and values are their counts.
    Example: "This is a test" --> {'This': 1, 'is': 1, 'a': 1, 'test': 1}
    """
    # Create a default dictionary to count occurrences
    wordCounts = collections.defaultdict(int)
    for word in inputString.split():
        wordCounts[word] += 1  # Increment count for each word
    return wordCounts

############################################################
# Task 3b: Stochastic Gradient Descent Implementation

def trainModel(trainingData, testingData, featureProcessor, iterations, learningRate):
    '''
    Train a model using stochastic gradient descent.

    @param trainingData: List of (input, label) tuples for training.
    @param testingData: List of (input, label) tuples for testing.
    @param featureProcessor: Function to extract features from input data.
    @param iterations: Number of training iterations.
    @param learningRate: Step size for updating weights.
    @return: A dictionary representing the trained weight vector.
    '''
    # Initialize weights as an empty dictionary
    weights = {}

    # Define a predictor function to classify input data
    def classify(inputData):
        return 1 if dotProduct(weights, featureProcessor(inputData)) > 0 else -1

    # Initialize weights for all features
    for data, label in trainingData:
        for feature in featureProcessor(data):
            weights[feature] = 0

    # Perform SGD for the specified number of iterations
    for epoch in range(iterations):
        for data, label in trainingData:
            # Update weights if prediction is incorrect
            if dotProduct(weights, featureProcessor(data)) * label < 1:
                increment(weights, learningRate * label, featureProcessor(data))
        # Evaluate performance (optional)
        # print(evaluatePredictor(testingData, classify))

    return weights

############################################################
# Task 3c: Dataset Generation for Testing

def createDataset(sampleCount, referenceWeights):
    '''
    Generate a dataset of examples consistent with the provided weights.

    @param sampleCount: Number of examples to generate.
    @param referenceWeights: Dictionary representing the weight vector.
    @return: List of (featureVector, label) tuples.
    '''
    random.seed(42)  # Ensure reproducibility

    def createSample():
        # Generate a feature vector with random values
        featureVector = {key: random.random() for key in random.sample(list(referenceWeights), len(referenceWeights) - 1)}
        # Determine label based on the weight vector
        label = 1 if dotProduct(referenceWeights, featureVector) > 0 else -1
        return (featureVector, label)

    return [createSample() for _ in range(sampleCount)]

############################################################
# Task 3e: Extracting Character Features

def getCharacterFeatures(n):
    '''
    Generate a function to extract n-gram character features from strings.

    @param n: The size of the n-grams.
    @return: A function to process input strings into n-gram feature vectors.
    '''
    def process(inputString):
        # Remove spaces and initialize a feature counter
        featureCounts = collections.defaultdict(int)
        cleanedString = inputString.replace(' ', '')

        # Extract n-grams
        for i in range(len(cleanedString) - n + 1):
            featureCounts[cleanedString[i:i+n]] += 1

        return featureCounts

    return process

############################################################
# Task 4: k-Means Clustering Algorithm
############################################################

def performKMeans(dataPoints, clusterCount, maxIterations):
    '''
    Perform k-means clustering on the provided data.

    @param dataPoints: List of examples (sparse vectors as dictionaries).
    @param clusterCount: Number of desired clusters.
    @param maxIterations: Maximum number of iterations for convergence.
    @return: Tuple (centroids, assignments, loss).
    '''
    def computeDistance(vectorA, vectorB):
        # Calculate squared distance between two sparse vectors
        return sum((vectorA.get(key, 0) - vectorB.get(key, 0))**2 for key in set(vectorA) | set(vectorB))

    # Initialize centroids randomly
    centroids = random.sample(dataPoints, clusterCount)
    assignments = [0] * len(dataPoints)

    for iteration in range(maxIterations):
        # Step 1: Assign each point to the closest centroid
        for idx, point in enumerate(dataPoints):
            minDistance = float('inf')
            for clusterIdx, centroid in enumerate(centroids):
                distance = computeDistance(point, centroid)
                if distance < minDistance:
                    minDistance = distance
                    assignments[idx] = clusterIdx

        # Step 2: Update centroids based on assignments
        for clusterIdx in range(clusterCount):
            aggregatedPoint = collections.defaultdict(float)
            count = assignments.count(clusterIdx)
            for idx, point in enumerate(dataPoints):
                if assignments[idx] == clusterIdx:
                    increment(aggregatedPoint, 1 / count, point)
            centroids[clusterIdx] = aggregatedPoint

    # Compute reconstruction loss
    reconstructionLoss = 0
    for idx, point in enumerate(dataPoints):
        diffVector = point.copy()
        increment(diffVector, -1, centroids[assignments[idx]])
        reconstructionLoss += dotProduct(diffVector, diffVector)

    return (centroids, assignments, reconstructionLoss)
