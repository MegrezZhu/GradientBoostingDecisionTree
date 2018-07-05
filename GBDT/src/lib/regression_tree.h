#pragma once

#include <vector>
#include <memory>

#include "util.h"

namespace zyuco {
	class RegressionTree {
	public:
		struct Config {
			double eta = 1.;				// shrinkage rate
			double gamma = 0.;				// minimum gain required to split a node
			size_t maxDepth = 6;			// max depth allowed
			size_t minChildWeight = 1;		// minimum allowed size for a node to be splitted
			size_t rounds = 1;				// number of subtrees
			double subsample = 1.;			// subsampling ratio for each tree
			double colsampleByTree = 1.;	// tree-wise feature subsampling ratio
		};
	protected:
		typedef std::vector<size_t> Index;
		typedef std::vector<Index> IndexMap;
		struct SplitPoint {
			size_t featureIndex;
			double splitPoint;
			double gain;
		};

		bool isLeaf = true;
		std::unique_ptr<RegressionTree> left, right;
		size_t featureIndex;
		double featureValue;

		double average; // predict value for leaf nodes

		RegressionTree();

		double predict(const Data::DataRow &r) const;

		static SplitPoint findSplitPoint(const Data::DataFrame &xx, const Data::DataColumn &y, const Index &index, const Index &featureIndexes);
		static double calculateError(size_t len, double sum, double powSum);
		static std::unique_ptr<RegressionTree> createNode(const Data::DataFrame &xx, const Data::DataColumn &y, const Index &index, const Index &featureIndexes, const Config &config, size_t leftDepth);

		// s: #random-samples, q: #bins
		static std::vector<double> sampleBinsDivider(const std::vector<double> &v, const Index &index, size_t s, size_t q);
		static size_t decideWhichBin(const std::vector<double> &divider, double value);
	public:
		Data::DataColumn predict(const Data::DataFrame &x) const;

		static std::unique_ptr<RegressionTree> fit(const Data::DataFrame &x, const Data::DataColumn &y, const Config &config);
	};
}
