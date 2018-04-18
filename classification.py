import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
from weka.core.dataset import Instances
import sys

def NaiveBayes(data):
	classifier = Classifier(classname="weka.classifiers.bayes.NaiveBayes")
	classifier.build_classifier(data)
	
	print("")
	print("=== Naive Bayes ===")
	print(classifier)

	count_class1 = 0
	count_class0 = 0
	print("Labeling income status of each instance. Please wait..")
	for index, inst in enumerate(data):
		pred = classifier.classify_instance(inst)
		# calculate no. of instances classified in class 1 and class 0
		if str(pred) == "1.0":
			count_class1 += 1
		else:
			count_class0 += 1
		if index%5000 == 0:
			print(".")

	print("No of instances in class '>50K' = "+ str(count_class1))
	print("No of instances in class '<=50K' = "+ str(count_class0))

def DecisionTree(data):

	classifier = Classifier(classname="weka.classifiers.trees.J48")
	classifier.build_classifier(data)

	print("")
	print("=== Decision Tree ===")
	print(classifier)

	count_class1 = 0
	count_class0 = 0
	print("Labeling income status of each instance. Please wait..")
	for index, inst in enumerate(data):
		pred = classifier.classify_instance(inst)
		# calculate no. of instances classified in class 1 and 0
		if str(pred) == "1.0":
			count_class1 += 1
		else:
			count_class0 += 1

		if index%5000 == 0:
			print(".")
		
	print("No of instances in class '>50K' = "+ str(count_class1))
	print("No of instances in class '<=50K' = "+ str(count_class0))

def main():

	try:
		jvm.start()

		loader = Loader(classname="weka.core.converters.CSVLoader")
		data = loader.load_file("./data/adult.csv") # load training data

		data.class_is_last() # set class attribute

		NaiveBayes(data)
		DecisionTree(data)
	except Exception as e:
		raise e
	finally:
		jvm.stop()

if __name__ == "__main__":
	main()