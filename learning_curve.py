import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
from weka.core.dataset import Instances
import sys
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

k = int(sys.argv[1])

def learning_curve(folds, data):
	training_set_size = []
	train_error_nb = []
	train_error_dtree = []
	cv_error_nb = []
	cv_error_dtree = []
	
	print("")
	print("This may take some time, please wait..")
	for training_size in range(10, 32561, 750):
		print(".")
		training_set_size.append(training_size)
		train_data = Instances.copy_instances(data, 0, training_size)

		data_size = train_data.num_instances
		fold_size = math.floor(data_size/folds)

		# calculating training and cross-validation error
		evaluation_nb_train = Evaluation(train_data)
		evaluation_nb_cv = Evaluation(train_data)
		evaluation_dtree_train = Evaluation(train_data)
		evaluation_dtree_cv = Evaluation(train_data)
		for i in range(folds):
			this_fold = fold_size
			test_start = i * fold_size
			test_end = (test_start + fold_size)
			if((data_size - test_end)/fold_size < 1):
				this_fold = data_size - test_start
			test = Instances.copy_instances(train_data, test_start, this_fold) # generate validation fold
			if i == 0:
				train = Instances.copy_instances(train_data, test_end, data_size - test_end)
			else:
				train_1 = Instances.copy_instances(train_data, 0, test_start)
				train_2 = Instances.copy_instances(train_data, test_end, data_size - test_end)
				train = Instances.append_instances(train_1, train_2) # generate training fold

			# Naive Bayes
			nb = Classifier(classname="weka.classifiers.bayes.NaiveBayes")
			nb.build_classifier(train)
			evaluation_nb_train.test_model(nb, train)
			evaluation_nb_cv.test_model(nb, test)

			# Decision Tree
			dtree = Classifier(classname="weka.classifiers.trees.J48")
			dtree.build_classifier(train)
			evaluation_dtree_train.test_model(dtree, train)
			evaluation_dtree_cv.test_model(dtree, test)

		train_error_nb.append(evaluation_nb_train.error_rate) # training error - NB
		cv_error_nb.append(evaluation_nb_cv.error_rate) # cross-validation error - NB
		train_error_dtree.append(evaluation_dtree_train.error_rate) # training error - DTree
		cv_error_dtree.append(evaluation_dtree_cv.error_rate) # cross-validation error - DTree

	# Plotting of Learning Curve
	x = training_set_size
	y1 = train_error_nb
	z1 = cv_error_nb
	y2 = train_error_dtree
	z2 = cv_error_dtree

	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13,8))

	axes[0].plot(x, y1, label='Training Error')
	axes[0].plot(x, z1, label='Cross-Validation Error')
	axes[0].set_xlabel('Training Set Size')
	axes[0].set_ylabel('Error Rate')
	axes[0].set_title('Naive Bayes')
	axes[0].legend()

	axes[1].plot(x, y2, label='Training Error')
	axes[1].plot(x, z2, label='Cross-Validation Error')
	axes[1].set_xlabel('Training Set Size')
	axes[1].set_ylabel('Error Rate')
	axes[1].set_title('Decision Tree')
	axes[1].legend()

	plt.show(block = True)


def main():

	try:
		jvm.start()

		loader = Loader(classname="weka.core.converters.CSVLoader")
		data = loader.load_file("./data/adult.csv")

		data.class_is_last() # set class attribute

		folds = k
		learning_curve(k, data)
	except Exception as e:
		raise e
	finally:
		jvm.stop()

if __name__ == "__main__":
	main()