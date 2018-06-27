#include <algorithm>
#include <iostream>

#include "GBDT.h"

using namespace std;

namespace zyuco {
	RegressionTree::RegressionTree() {
	}

	double RegressionTree::predict(const Data::DataRow &r) const {
		if (isLeaf) return average;
		if (r[featureIndex] <= featureValue) return left->predict(r);
		else return right->predict(r);
	}

	RegressionTree::SplitPoint RegressionTree::findSplitPoint(const Data::DataFrame &x, const Data::DataColumn &y, const Index &index) {
		double bestGain = 0, bestSplit = 0;
		size_t bestFeature = 0;

		// TODO: parallel
		for (size_t featureIndex = 0; featureIndex < x.front().size(); featureIndex++) {
			vector<tuple<size_t, double, double>> v;
			v.reserve(index.size());

			for (auto ind : index) {
				v.push_back({ ind, x[ind][featureIndex], y[ind] });
			}

			tuple<size_t, double, double> tup;
			// TODO: use index map & pre-sorting instead
			sort(v.begin(), v.end(), [](const auto &l, const auto &r) {
				return get<1>(l) < get<1>(r);
			});

			double wholeErr, leftErr, rightErr;
			double wholeSum = 0, leftSum, rightSum;
			double wholePowSum = 0, leftPowSum, rightPowSum;
			for (const auto &t : v) {
				wholeSum += get<2>(t);
				wholePowSum += pow(get<2>(t), 2);
			}
			wholeErr = calculateError(index.size(), wholeSum, wholePowSum);

			leftSum = leftPowSum = 0;
			rightSum = wholeSum;
			rightPowSum = wholePowSum;
			for (size_t i = 0; i + 1 < index.size(); i++) {
				auto label = get<2>(v[i]);

				leftSum += label;
				rightSum -= label;
				leftPowSum += pow(label, 2);
				rightPowSum -= pow(label, 2);

				if (get<2>(v[i]) == get<2>(v[i + 1])) continue; // same label with next, not splitable
				if (get<1>(v[i]) == get<1>(v[i + 1])) continue; // same value, not splitable

				leftErr = calculateError(i + 1, leftSum, leftPowSum);
				rightErr = calculateError(index.size() - i - 1, rightSum, rightPowSum);

				double gain = wholeErr - ((i + 1) * leftErr / index.size() + (index.size() - i - 1) * rightErr / index.size());
				if (gain > bestGain) {
					bestGain = gain;
					bestSplit = (get<1>(v[i]) + get<1>(v[i + 1])) / 2;
					bestFeature = featureIndex;
				}
			}
		}

		cout << "best gain: " << bestGain << endl;
		cout << "split at: " << bestSplit << endl;
		cout << "on feature " << bestFeature << endl << endl;

		return { bestFeature,bestSplit,bestGain };
	}

	double RegressionTree::calculateError(size_t len, double sum, double powSum) {
		double avg = sum / len;
		double a = powSum / len;
		double b = pow(avg, 2);
		double c = 2. * avg * sum / len;
		return a + b - c;
	}

	std::unique_ptr<RegressionTree> RegressionTree::createNode(const Data::DataFrame &x, const Data::DataColumn &y, const Index &index, int maxDepth) {
		// depth-first growing

		if (maxDepth <= 0) return unique_ptr<RegressionTree>();

		auto p = new RegressionTree();

		// calculate value for prediction
		p->average = 0;
		for (auto ind : index) p->average += y[ind];
		p->average /= index.size();

		if (x.size() > 1) {
			// try to split
			auto ret = findSplitPoint(x, y, index);
			if (ret.gain > 0 && maxDepth > 1) { // check splitablity
				// split points
				p->isLeaf = false;
				vector<size_t> leftIndex, rightIndex;
				for (auto ind : index) {
					if (x[ind][ret.featureIndex] <= ret.splitPoint) {
						leftIndex.push_back(ind); // to the left
					}
					else {
						rightIndex.push_back(ind); // to the right
					}
				}
				if (leftIndex.size() == 0 || rightIndex.size() == 0) throw runtime_error("unexpected empty subtree");
				p->featureIndex = ret.featureIndex;
				p->featureValue = ret.splitPoint;

				p->left = createNode(x, y, leftIndex, maxDepth - 1);
				p->right = createNode(x, y, rightIndex, maxDepth - 1);
			}
		}

		return unique_ptr<RegressionTree>(p);
	}

	Data::DataColumn RegressionTree::predict(const Data::DataFrame & x) const {
		Data::DataColumn result;
		result.reserve(x.size());
		for (const auto &r : x) result.push_back(predict(r));
		return result;
	}

	std::unique_ptr<RegressionTree> RegressionTree::fit(const Data::DataFrame &x, const Data::DataColumn &y, int maxDepth) {
		if (x.size() != y.size()) throw invalid_argument("x, y has different size");
		if (x.empty() || y.empty()) throw invalid_argument("empty dataset");
		vector<size_t> index(y.size());
		for (size_t i = 0; i < y.size(); i++) index[i] = i;
		return createNode(x, y, index, maxDepth);
	}
}
