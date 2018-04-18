import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
from weka.core.dataset import Instances

def testNB(training_data, testing_data):

	train_data = Instances.copy_instances(training_data)
	test_data = Instances.copy_instances(testing_data)

	evaluation = Evaluation(train_data)
	classifier = Classifier(classname="weka.classifiers.bayes.NaiveBayes")
	classifier.build_classifier(train_data) # build classifier on the training data
	evaluation.test_model(classifier, test_data) # test and evaluate model on the test set
	print("")
	print("")
	print(evaluation.summary("--------------Naive Bayes Evaluation--------------"))
	print("Accuracy: " + str(evaluation.percent_correct))
	print("")
	print("Label\tPrecision\t\tRecall\t\t\tF-Measure")
	print("<=50K\t"+str(evaluation.precision(0))+"\t"+str(evaluation.recall(0))+"\t"+str(evaluation.f_measure(0)))
	print(">50K\t"+str(evaluation.precision(1))+"\t"+str(evaluation.recall(1))+"\t"+str(evaluation.f_measure(1)))
	print("Mean\t"+str(((evaluation.precision(1))+(evaluation.precision(0)))/2)+"\t"+str(((evaluation.recall(1))+(evaluation.recall(0)))/2)+"\t"+str(((evaluation.f_measure(1))+(evaluation.f_measure(0)))/2))
	
def testDtree(training_data, testing_data):
	train_data = Instances.copy_instances(training_data)
	test_data = Instances.copy_instances(testing_data)

	evaluation = Evaluation(train_data)
	classifier = Classifier(classname="weka.classifiers.trees.J48")
	classifier.build_classifier(train_data) # build classifier on the training data
	evaluation.test_model(classifier, test_data) # test and evaluate model on the test set
	print("")
	print("")
	print(evaluation.summary("--------------Decision Tree Evaluation--------------"))
	print("Accuracy: " + str(evaluation.percent_correct))
	print("")
	print("Label\tPrecision\t\tRecall\t\t\tF-Measure")
	print("<=50K\t"+str(evaluation.precision(0))+"\t"+str(evaluation.recall(0))+"\t"+str(evaluation.f_measure(0)))
	print(">50K\t"+str(evaluation.precision(1))+"\t"+str(evaluation.recall(1))+"\t"+str(evaluation.f_measure(1)))
	print("Mean\t"+str(((evaluation.precision(1))+(evaluation.precision(0)))/2)+"\t"+str(((evaluation.recall(1))+(evaluation.recall(0)))/2)+"\t"+str(((evaluation.f_measure(1))+(evaluation.f_measure(0)))/2))

def main():

	try:
		jvm.start()

		loader = Loader(classname="weka.core.converters.CSVLoader")
		training_data = loader.load_file("./data/adult.csv") # load training set
		testing_data = loader.load_file("./data/adult_test.csv") # load test set

		training_data.class_is_last()
		testing_data.class_is_last()

		testNB(training_data, testing_data)
		testDtree(training_data, testing_data)
	except Exception as e:
		raise e
	finally:
		jvm.stop()

if __name__ == "__main__":
	main()
