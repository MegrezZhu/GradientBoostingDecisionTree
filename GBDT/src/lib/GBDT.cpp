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

	RegressionTree::SplitPoint RegressionTree::findSplitPoint(const Data::DataFrame &xx, const Data::DataColumn &y, const Index &index) {
		double bestGain = 0, bestSplit = 0;
		size_t bestFeature = 0;
		size_t num = index.size();

		// TODO: parallel
		#pragma omp parallel for
		for (int i = 0; i < xx.size(); i++) {
			size_t featureIndex = i;

			vector<double> featureValues(num);
			for (size_t i = 0; i < index.size(); i++) {
				featureValues[i] = xx[featureIndex][index[i]]; // for cache
			}
			size_t nSample = pow(num, .5), nBin = pow(num, .25);
			auto dividers = sampleBinsDivider(featureValues, nSample, nBin);
			vector<double> binSums(nBin, .0), binPowSums(nBin, .0);
			vector<size_t> binSizes(nBin, 0);
			for (size_t i = 0; i < num; i++) {
				auto into = decideWhichBin(dividers, featureValues[i]);
				auto label = y[index[i]];
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

	std::unique_ptr<RegressionTree> RegressionTree::createNode(const Data::DataFrame &xx, const Data::DataColumn &y, const Index &index, const BoostingConfig &config, size_t leftDepth) {
		// depth-first growing

		if (leftDepth <= 0) return unique_ptr<RegressionTree>();

		auto p = new RegressionTree();

		// calculate value for prediction
		p->average = 0.;
		for (auto ind : index) p->average += y[ind];
		p->average /= index.size();

		if (index.size() > max<size_t>(1, config.minChildWeight)) {
			// try to split
			auto ret = findSplitPoint(xx, y, index);
			if (ret.gain > config.gamma && leftDepth > 1) { // check splitablity
				// split points
				vector<size_t> leftIndex, rightIndex;
				for (auto ind : index) {
					if (xx[ret.featureIndex][ind] <= ret.splitPoint) {
						leftIndex.push_back(ind); // to the left
					}
					else {
						rightIndex.push_back(ind); // to the right
					}
				}
				// if (leftIndex.size() == 0 || rightIndex.size() == 0) throw runtime_error("unexpected empty subtree");
				if (leftIndex.size() != 0 && rightIndex.size() != 0) {
					p->isLeaf = false;

					p->featureIndex = ret.featureIndex;
					p->featureValue = ret.splitPoint;

					p->left = createNode(xx, y, leftIndex, config, leftDepth - 1);
					p->right = createNode(xx, y, rightIndex, config, leftDepth - 1);

					cout << NOW << "node split at feature " << ret.featureIndex << " with gain " << ret.gain << '\n';
				}
			}
		}

		return unique_ptr<RegressionTree>(p);
	}

	std::vector<double> RegressionTree::sampleBinsDivider(const std::vector<double>& v, size_t s, size_t q) {
		vector<double> samples(s);
		std::random_device rd;
		auto gen = std::default_random_engine(rd());
		std::uniform_int_distribution<int> dis(0, v.size() - 1);
		for (size_t i = 0; i < s; i++) samples[i] = v[dis(gen)];

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
		return result;
	}

	std::unique_ptr<RegressionTree> RegressionTree::fit(const Data::DataFrame &xx, const Data::DataColumn &y, const BoostingConfig &config) {
		Index index(xx.front().size());
		for (size_t i = 0; i < index.size(); i++) index[i] = i;

		return createNode(xx, y, index, config, config.maxDepth);
	}

	Data::DataColumn GradientBoostingClassifer::predict(const Data::DataFrame & x) const {
		Data::DataColumn result(x.size(), 0.);
		for (const auto& ptr : trees) {
			auto subResult = ptr->predict(x);
			subResult *= config.eta;
			result += subResult; // better cache performance ?
		}
		return result;
	}

	std::unique_ptr<GradientBoostingClassifer> GradientBoostingClassifer::fit(const Data::DataFrame & x, const Data::DataColumn & y, const BoostingConfig &config) {
		auto p = new GradientBoostingClassifer();

		if (x.size() != y.size()) throw invalid_argument("x, y has different size");
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
		auto roundsLeft = config.rounds;
		while (roundsLeft--) {
			auto subtree = RegressionTree::fit(xx, residual, config);

			auto pred = subtree->predict(x);
			pred *= config.eta;
			residual -= pred;

			p->trees.push_back(move(subtree));
			cout << NOW << config.rounds - roundsLeft << "th round finished\n";

			auto totalPred = y;
			totalPred -= residual;
			cout << NOW << "training accuracy: " << calculateAccuracy(totalPred, y) << ", training auc: " << calculateAUC(totalPred, y) << endl;
		}

		return unique_ptr<GradientBoostingClassifer>(p);
	}
}
