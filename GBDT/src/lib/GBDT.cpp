#include <algorithm>
#include <omp.h>
#include <iostream>

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

		// TODO: parallel
		#pragma omp parallel for
		for (int i = 0; i < xx.size(); i++) {
			size_t featureIndex = i;
			vector<pair<size_t, double>> v(index.size());

			for (size_t i = 0; i < index.size(); i++) {
				auto ind = index[i];
				v[i].first = ind;
				v[i].second = xx[featureIndex][ind];
			}

			tuple<size_t, double, double> tup;
			// TODO: use index map & pre-sorting instead
			sort(v.begin(), v.end(), [](const auto &l, const auto &r) {
				return l.second < r.second;
			});

			double wholeErr, leftErr, rightErr;
			double wholeSum = 0, leftSum, rightSum;
			double wholePowSum = 0, leftPowSum, rightPowSum;
			for (const auto &t : v) {
				wholeSum += y[t.first];
				wholePowSum += pow(y[t.first], 2);
			}
			wholeErr = calculateError(index.size(), wholeSum, wholePowSum);

			leftSum = leftPowSum = 0;
			rightSum = wholeSum;
			rightPowSum = wholePowSum;
			for (size_t i = 0; i + 1 < index.size(); i++) {
				auto label = y[v[i].first];

				leftSum += label;
				rightSum -= label;
				leftPowSum += pow(label, 2);
				rightPowSum -= pow(label, 2);

				if (y[v[i].first] == y[v[i + 1].first]) continue; // same label with next, not splitable
				if (v[i].second == v[i + 1].second) continue; // same value, not splitable

				leftErr = calculateError(i + 1, leftSum, leftPowSum);
				rightErr = calculateError(index.size() - i - 1, rightSum, rightPowSum);

				double gain = wholeErr - ((i + 1) * leftErr / index.size() + (index.size() - i - 1) * rightErr / index.size());

				#pragma omp critical
				if (gain > bestGain) {
					bestGain = gain;
					bestSplit = (v[i].second + v[i + 1].second) / 2;
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
				p->isLeaf = false;
				vector<size_t> leftIndex, rightIndex;
				for (auto ind : index) {
					if (xx[ret.featureIndex][ind] <= ret.splitPoint) {
						leftIndex.push_back(ind); // to the left
					}
					else {
						rightIndex.push_back(ind); // to the right
					}
				}
				if (leftIndex.size() == 0 || rightIndex.size() == 0) throw runtime_error("unexpected empty subtree");
				p->featureIndex = ret.featureIndex;
				p->featureValue = ret.splitPoint;

				p->left = createNode(xx, y, leftIndex, config, leftDepth - 1);
				p->right = createNode(xx, y, rightIndex, config, leftDepth - 1);

				cout << NOW << "node split at feature " << ret.featureIndex << " with gain " << ret.gain << '\n';
			}
		}

		return unique_ptr<RegressionTree>(p);
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
		//// initializing indexMap
		//IndexMap indexMap(x.front().size(), Index(x.size())); // size: nFeature * nSample
		//for (size_t i = 0; i < y.size(); i++) {
		//	for (auto &index : indexMap) index[i] = i;
		//}

		//// pre-sorting
		//for (size_t featureIndex = 0; featureIndex < x.front().size(); featureIndex++) {
		//	sort(indexMap[featureIndex].begin(), indexMap[featureIndex].end(), [](const auto &a, const auto &b) {
		//		return xx[featureIndex][a] < xx[featureIndex][b];
		//	});
		//}
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
		}

		return unique_ptr<GradientBoostingClassifer>(p);
	}
}
