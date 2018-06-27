#pragma once

#include <vector>
#include <memory>
#include <tuple>

#include "util.h"

namespace zyuco {
	class RegressionTree;

	class GradientBoostingClassifer {
        struct Config {
            double eta;
			size_t maxDepth;
			// below are not supported yet
			double minChildWeight;
			double minimumGain;
        };

		std::vector<std::unique_ptr<RegressionTree>> trees;

	public:
		Data::DataColumn predict(const Data::DataFrame &x) const;

		static std::unique_ptr<GradientBoostingClassifer> fit(const Data::DataFrame &x, const Data::DataColumn &y, size_t maxDepth, size_t iters);
	};

	class RegressionTree {
		typedef std::vector<size_t> Index;
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

		static SplitPoint findSplitPoint(const Data::DataFrame &x, const Data::DataColumn &y, const Index &index);
		static double calculateError(size_t len, double sum, double powSum);
		static std::unique_ptr<RegressionTree> createNode(const Data::DataFrame &x, const Data::DataColumn &y, const Index &index, int maxDepth);
	public:
		Data::DataColumn predict(const Data::DataFrame &x) const;

		static std::unique_ptr<RegressionTree> fit(const Data::DataFrame &x, const Data::DataColumn &y, int maxDepth);
	};
}
