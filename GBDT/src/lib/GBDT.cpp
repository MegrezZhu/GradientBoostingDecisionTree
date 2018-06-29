#include <algorithm>
#include <omp.h>
#include <iostream>
#include <random>

#include "GBDT.h"
#include "logger.h"

using namespace std;

namespace zyuco {
	RegressionTree::RegressionTree() {
	}

	double RegressionTree::predict(const Data::DataRow &r) const {
		if (isLeaf) return average;
		if (r[featureIndex] <= featureValue) return left->predict(r);
		else return right->predict(r);
	}

	RegressionTree::SplitPoint RegressionTree::findSplitPoint(const Data::DataFrame &xx, const Data::DataColumn &y, const Index &index, const Index &featureIndexes) {
		double bestGain = 0, bestSplit = 0;
		size_t bestFeature = 0;
		size_t num = index.size();

		#pragma omp parallel for
		for (int i = 0; i < featureIndexes.size(); i++) {
			size_t featureIndex = featureIndexes[i];

			size_t nSample = size_t(pow(num, .5)), nBin = size_t(pow(num, .25));
			auto dividers = sampleBinsDivider(xx[featureIndex], index, nSample, nBin);
			vector<double> binSums(nBin, .0), binPowSums(nBin, .0);
			vector<size_t> binSizes(nBin, 0);
			for (int i = 0; i < num; i++) {
				auto value = xx[featureIndex][index[i]];
				auto into = decideWhichBin(dividers, value);
				auto label = y[i];
				binSums[into] += label;
				binPowSums[into] += pow(label, 2);
				binSizes[into]++;
			}

			double wholeErr, leftErr, rightErr;
			double wholeSum = 0, leftSum, rightSum;
			double wholePowSum = 0, leftPowSum, rightPowSum;
			size_t leftSize = 0, rightSize = num;
			for (auto t : binSums) wholeSum += t;
			for (auto t : binPowSums) wholePowSum += t;
			wholeErr = calculateError(num, wholeSum, wholePowSum);

			leftSum = leftPowSum = 0;
			rightSum = wholeSum;
			rightPowSum = wholePowSum;
			for (size_t i = 0; i + 1 < binSums.size(); i++) {
				auto divider = dividers[i];
				auto binSum = binSums[i];
				auto binPowSum = binPowSums[i];
				auto binSize = binSizes[i];

				leftSum += binSum;
				rightSum -= binSum;
				leftPowSum += binPowSum;
				rightPowSum -= binPowSum;
				leftSize += binSize;
				rightSize -= binSize;

				leftErr = calculateError(leftSize, leftSum, leftPowSum);
				rightErr = calculateError(rightSize, rightSum, rightPowSum);

				double gain = wholeErr - (leftSize * leftErr / num + rightSize * rightErr / num);

				#pragma omp critical
				if (gain > bestGain) {
					bestGain = gain;
					bestSplit = divider;
					bestFeature = featureIndex;
				}
			}
		}

		return { bestFeature, bestSplit, bestGain };
	}

	double RegressionTree::calculateError(size_t len, double sum, double powSum) {
		double avg = sum / len;
		double a = powSum / len;
		double b = pow(avg, 2);
		double c = 2. * avg * sum / len;
		return a + b - c;
	}

	std::unique_ptr<RegressionTree> RegressionTree::createNode(const Data::DataFrame &xx, const Data::DataColumn &y, const Index &index, const Index &featureIndexes, const BoostingConfig &config, size_t leftDepth) {
		// depth-first growing

		if (leftDepth <= 0) return unique_ptr<RegressionTree>();

		auto p = new RegressionTree();

		// calculate value for prediction
		p->average = 0.;
		for (auto v : y) p->average += v;
		p->average /= index.size();

		if (index.size() > max<size_t>(1, config.minChildWeight)) {
			// try to split
			auto ret = findSplitPoint(xx, y, index, featureIndexes);
			if (ret.gain > config.gamma && leftDepth > 1) { // check splitablity
				// split points
				vector<size_t> leftIndex, rightIndex;
				Data::DataColumn leftY, rightY;
				for (size_t i = 0; i < index.size(); i++) {
					auto ind = index[i];
					if (xx[ret.featureIndex][ind] <= ret.splitPoint) {
						leftIndex.push_back(ind); // to the left
						leftY.push_back(y[i]);
					}
					else {
						rightIndex.push_back(ind); // to the right
						rightY.push_back(y[i]);
					}
				}
				// if (leftIndex.size() == 0 || rightIndex.size() == 0) throw runtime_error("unexpected empty subtree");
				if (leftIndex.size() != 0 && rightIndex.size() != 0) {
					p->isLeaf = false;

					p->featureIndex = ret.featureIndex;
					p->featureValue = ret.splitPoint;

					p->left = createNode(xx, leftY, leftIndex, featureIndexes, config, leftDepth - 1);
					p->right = createNode(xx, rightY, rightIndex, featureIndexes, config, leftDepth - 1);

					//cout << NOW << "node split at feature " << ret.featureIndex << " with gain " << ret.gain << '\n';
				}
			}
		}

		return unique_ptr<RegressionTree>(p);
	}

	std::vector<double> RegressionTree::sampleBinsDivider(const std::vector<double>& v, const Index & index, size_t s, size_t q) {
		vector<double> samples(s);
		std::random_device rd;
		auto gen = std::default_random_engine(rd());
		std::uniform_int_distribution<size_t> dis(0, index.size() - 1);
		for (size_t i = 0; i < s; i++) samples[i] = v[index[dis(gen)]];

		sort(samples.begin(), samples.end());

		vector<double> divider(q - 1);
		size_t space = (samples.size() - (q - 1)) / q;
		for (size_t i = 0; i < q - 1; i++) {
			divider[i] = samples[(space + 1) * i];
		}
		return divider;
	}

	size_t RegressionTree::decideWhichBin(const std::vector<double>& divider, double value) {
		// linear search significantly outperforms binary search, why?
		// due to the mostly zeros?

		//int l = 0, r = divider.size() - 1;
		//size_t result = r + 1;
		//while (l <= r) {
		//	size_t m = (l + r) / 2;
		//	if (value <= divider[m]) {
		//		result = m;
		//		r = m - 1;
		//	}
		//	else {
		//		l = m + 1;
		//	}
		//}
		//return result;

		for (size_t i = 0; i < divider.size(); i++) {
			if (value <= divider[i]) return i;
		}
		return divider.size();
	}

	Data::DataColumn RegressionTree::predict(const Data::DataFrame & x) const {
		Data::DataColumn result(x.size());
		#pragma omp parallel for
		for (int i = 0; i < x.size(); i++) {
			result[i] = predict(x[i]);
		}
		// trim outliners
		for (auto &v : result) {
			v = max(v, .0);
			v = min(v, 1.);
		}
		return result;
	}

	std::unique_ptr<RegressionTree> RegressionTree::fit(const Data::DataFrame &xx, const Data::DataColumn &y, const BoostingConfig &config) {
		std::random_device rd;
		auto gen = std::default_random_engine(rd());

		// generate subsample
		auto sampleSize = size_t(y.size() * config.subsample);
		Index index(sampleSize);
		Data::DataColumn sampledY(sampleSize);
		std::uniform_int_distribution<size_t> dis(0, y.size() - 1);
		for (size_t i = 0; i < index.size(); i++) index[i] = dis(gen); // sample with replacement
		sort(index.begin(), index.end()); // for cache
		for (size_t i = 0; i < index.size(); i++) sampledY[i] = y[index[i]];

		// generate colsample
		auto colsampleSize = size_t(xx.size() * config.colsampleByTree);
		Index featureIndexes(xx.size());
		for (size_t i = 0; i < featureIndexes.size(); i++) featureIndexes[i] = i;
		shuffle(featureIndexes.begin(), featureIndexes.end(), mt19937(rd()));
		featureIndexes.resize(colsampleSize); // sample without replacement
		sort(featureIndexes.begin(), featureIndexes.end()); // for cache

		return createNode(xx, sampledY, index, featureIndexes, config, config.maxDepth);
	}

	Data::DataColumn GradientBoostingClassifier::predict(const Data::DataFrame & x) const {
		Data::DataColumn result(x.size(), 0.);
		for (const auto& ptr : trees) {
			auto subResult = ptr->predict(x);
			subResult *= config.eta;
			result += subResult; // better cache performance ?
		}
		return result;
	}

	std::unique_ptr<GradientBoostingClassifier> GradientBoostingClassifier::fit(const Data::DataFrame &x, const Data::DataColumn &y, const BoostingConfig &config, const Data::DataFrame &tx, const Data::DataColumn &ty) {
		auto p = new GradientBoostingClassifier();

		if (x.size() != y.size()) throw invalid_argument("in training data, x y has different length");
		if (tx.size() != ty.size()) throw invalid_argument("in validation data, x y has different length");
		if (x.empty() || y.empty()) throw invalid_argument("empty dataset");

		p->config = config;

		// reshaping input x into a column-first nFeature * nSample matrix
		// for better cache performance
		Data::DataFrame xx(x.front().size(), Data::DataRow(x.size()));
		for (size_t i = 0; i < x.size(); i++) {
			const auto &row = x[i];
			for (size_t j = 0; j < row.size(); j++) xx[j][i] = row[j];
		}

		auto residual = y;
		Data::DataColumn validationPred(ty.size(), 0.);
		auto roundsLeft = config.rounds;
		while (roundsLeft--) {
			auto subtree = RegressionTree::fit(xx, residual, config);

			auto pred = subtree->predict(x);
			pred *= config.eta;
			residual -= pred;

			cout << NOW << config.rounds - roundsLeft << "th round finished\n";

			auto totalPred = y;
			totalPred -= residual;
			cout << NOW << "training accuracy: " << calculateAccuracy(totalPred, y) << ", training auc: " << calculateAUC(totalPred, y) << endl;

			// if validation set is non-empty, then
			if (!tx.empty()) {
				auto validSubPred = subtree->predict(tx);
				validSubPred *= config.eta;
				validationPred += validSubPred;
				cout << NOW << "validation accuracy: " << calculateAccuracy(validationPred, ty) << ", validation auc: " << calculateAUC(validationPred, ty) << endl;
			}

			p->trees.push_back(move(subtree));
		}

		return unique_ptr<GradientBoostingClassifier>(p);
	}
}
