#pragma once

#include <vector>
#include <memory>
#include <tuple>

#include "util.h"

namespace zyuco {
	struct BoostingConfig {
		double eta = 1.;				// shrinkage rate
		double gamma = 0.;				// minimum gain required to split a node
		size_t maxDepth = 6;			// max depth allowed
		size_t minChildWeight = 1;		// minimum allowed size for a node to be splitted
		size_t rounds = 1;				// number of subtrees
	};

	class RegressionTree;

	class GradientBoostingClassifer {
		BoostingConfig config;
		std::vector<std::unique_ptr<RegressionTree>> trees;
	public:

		Data::DataColumn predict(const Data::DataFrame &x) const;

		static std::unique_ptr<GradientBoostingClassifer> fit(const Data::DataFrame &x, const Data::DataColumn &y, const BoostingConfig &config);
	};

	class RegressionTree {
		typedef std::vector<size_t> Index;
		typedef std::vector<Index> IndexMap;
	protected:
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

		static SplitPoint findSplitPoint(const Data::DataFrame &xx, const Data::DataColumn &y, const Index &index);
		static double calculateError(size_t len, double sum, double powSum);
		static std::unique_ptr<RegressionTree> createNode(const Data::DataFrame &xx, const Data::DataColumn &y, const Index &index, const BoostingConfig &config, size_t leftDepth);

		// s: #random-samples, q: #bins
		static std::vector<double> sampleBinsDivider(const std::vector<double> &v, size_t s, size_t q);
		static std::vector<double> sampleBinsDivider(const std::vector<double> &v, const Index &index, size_t s, size_t q);
		static size_t decideWhichBin(const std::vector<double> &divider, double value);
	public:
		Data::DataColumn predict(const Data::DataFrame &x) const;

		static std::unique_ptr<RegressionTree> fit(const Data::DataFrame &x, const Data::DataColumn &y, const BoostingConfig &config);
	};
}
