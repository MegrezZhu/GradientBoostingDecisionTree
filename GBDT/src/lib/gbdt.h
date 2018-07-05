#pragma once

#include <vector>
#include <memory>

#include "util.h"
#include "regression_tree.h"

namespace zyuco {
	class GradientBoostingClassifier {
	public:
		typedef RegressionTree::Config Config;

		static std::unique_ptr<GradientBoostingClassifier> fit(const Data::DataFrame &x, const Data::DataColumn &y, const Config &config, const Data::DataFrame &tx = Data::DataFrame(), const Data::DataColumn &ty = Data::DataColumn());

		Data::DataColumn predict(const Data::DataFrame &x) const;
	protected:
		Config config;
		std::vector<std::unique_ptr<RegressionTree>> trees;
	};
}
