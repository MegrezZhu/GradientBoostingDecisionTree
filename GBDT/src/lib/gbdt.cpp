#include <algorithm>
#include <omp.h>
#include <iostream>
#include <random>

#include "gbdt.h"
#include "logger.h"

using namespace std;

namespace zyuco {
	Data::DataColumn GradientBoostingClassifier::predict(const Data::DataFrame & x) const {
		Data::DataColumn result(x.size(), 0.);
		for (const auto& ptr : trees) {
			auto subResult = ptr->predict(x);
			subResult *= config.eta;
			result += subResult; // better cache performance ?
		}
		// trim outliners
		for (auto &v : result) {
			v = max(v, .0);
			v = min(v, 1.);
		}
		return result;
	}

	std::unique_ptr<GradientBoostingClassifier> GradientBoostingClassifier::fit(const Data::DataFrame &x, const Data::DataColumn &y, const Config &config, const Data::DataFrame &tx, const Data::DataColumn &ty) {
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
			auto start = getCurrentMillisecond();
			auto subtree = RegressionTree::fit(xx, residual, config);

			auto pred = subtree->predict(x);
			pred *= config.eta;
			residual -= pred;

			cout << NOW << config.rounds - roundsLeft << "th round finished, used " << getCurrentMillisecond() - start << "ms" << endl;

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
