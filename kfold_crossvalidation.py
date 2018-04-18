import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
from weka.core.dataset import Instances
import sys
import math

k = int(sys.argv[1])

def NaiveBayes(rnd_data, folds, seed, data):

	data_size = rnd_data.num_instances
	fold_size = math.floor(data_size/folds)

	# cross-validation
	evaluation = Evaluation(rnd_data)
	for i in range(folds):
		this_fold = fold_size
		test_start = i * fold_size
		test_end = (test_start + fold_size)
		if((data_size - test_end)/fold_size < 1):
			this_fold = data_size - test_start
		test = Instances.copy_instances(rnd_data, test_start, this_fold) # generate validation fold
		if i == 0:
			train = Instances.copy_instances(rnd_data, test_end, data_size - test_end)
		else:
			train_1 = Instances.copy_instances(rnd_data, 0, test_start)
			train_2 = Instances.copy_instances(rnd_data, test_end, data_size - test_end)
			train = Instances.append_instances(train_1, train_2) # generate training fold
		
		# build and evaluate classifier
		cls = Classifier(classname="weka.classifiers.bayes.NaiveBayes")
		cls.build_classifier(train) # build classifier on training set
		evaluation.test_model(cls, test) # test classifier on validation/test set

	print("")
	print("=== Naive Bayes ===")
	print("Classifier: " + cls.to_commandline())
	print("Dataset: " + data.relationname)
	print("Folds: " + str(folds))
	print("Seed: " + str(seed))
	print("")
	print(evaluation.summary("=== " + str(folds) + "-fold Cross-Validation ==="))

def DecisionTree(rnd_data, folds, seed, data):

	data_size = rnd_data.num_instances
	fold_size = math.floor(data_size/folds)

	# cross-validation
	evaluation = Evaluation(rnd_data)
	for i in range(folds):
		this_fold = fold_size
		test_start = i * fold_size
		test_end = (test_start + fold_size)
		if((data_size - test_end)/fold_size < 1):
			this_fold = data_size-test_start
		test = Instances.copy_instances(rnd_data, test_start, this_fold) # generate validation fold
		if i == 0:
			train = Instances.copy_instances(rnd_data, test_end, data_size - test_end)
		else:
			train_1 = Instances.copy_instances(rnd_data, 0, test_start)
			train_2 = Instances.copy_instances(rnd_data, test_end, data_size - test_end)
			train = Instances.append_instances(train_1, train_2) # generate training fold

		# build and evaluate classifier
		cls = Classifier(classname="weka.classifiers.trees.J48")
		cls.build_classifier(train) # build classifier on training set
		evaluation.test_model(cls, test) # test classifier on validation/test set

	print("")
	print("=== Decision Tree ===")
	print("Classifier: " + cls.to_commandline())
	print("Dataset: " + data.relationname)
	print("Folds: " + str(folds))
	print("Seed: " + str(seed))
	print("")
	print(evaluation.summary("=== " + str(folds) + "-fold Cross-Validation ==="))

def main():

	try:
		jvm.start()

		loader = Loader(classname="weka.core.converters.CSVLoader")
		data = loader.load_file("./data/adult.csv")

		data.class_is_last() # set class attribute

		# randomize data
		folds = k
		seed = 1
		rnd = Random(seed)
		rand_data = Instances.copy_instances(data)
		rand_data.randomize(rnd)
		if rand_data.class_attribute.is_nominal:
			rand_data.stratify(folds)

		NaiveBayes(rand_data, folds, seed, data)
		DecisionTree(rand_data, folds, seed, data)
	except Exception as e:
		raise e
	finally:
		jvm.stop()

if __name__ == "__main__":
	main()