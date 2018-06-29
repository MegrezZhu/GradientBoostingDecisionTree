#include <fstream>
#include <algorithm>
#include <cctype>
#include <sstream>
#include <chrono>
#include <iostream>
#include <random>
#include <omp.h>

#include "util.h"
#include "logger.h"

using namespace std;

namespace zyuco {
	namespace Data {
		LibSVMData fromLibSVM(const std::string & path, size_t featureCount) {
			DataFrame x;
			DataColumn y;

			string content;
			getline(ifstream(path), content, '\0');
			stringstream in(move(content));

			vector<string> lines;
			string line;
			while (getline(in, line)) lines.push_back(move(line));

			#pragma omp parallel for
			for (int i = 0; i < lines.size(); i++) {
				auto item = parseLibSVMLine(move(lines[i]), featureCount);
				#pragma omp critical
				{
					x.push_back(move(item.first));
					y.push_back(item.second);
				}
			}

			return { move(x), move(y) };
		}

		void toCSV(const DataColumn & ids, const DataColumn & pred, const std::vector<size_t> &index, const std::string & path) {
			ofstream out(path);
			if (!out.is_open()) throw runtime_error("cannot open " + path);
			out << "id,label" << endl; // header
			for (size_t i = 0; i < ids.size(); i++) {
				out << ids[index[i]] << ',' << pred[index[i]] << endl;
			}
		}

		SplitXYResult splitXY(DataFrame &&x, DataColumn &&y, double trainSize) {
			size_t total = x.size();
			size_t trainCount = size_t(total * trainSize), testCount = total - trainCount;

			// shuffle
			vector<size_t> indices(total);
			for (size_t i = 0; i < total; i++) indices[i] = i;
			shuffle(indices.begin(), indices.end(), mt19937(random_device()()));

			DataFrame trainX, testX;
			DataColumn trainY, testY;
			trainX.reserve(trainCount);
			trainY.reserve(trainCount);
			testX.reserve(testCount);
			testY.reserve(testCount);
			for (size_t i = 0; i < trainCount; i++) {
				trainX.push_back(move(x[i]));
				trainY.push_back(move(y[i]));
			}
			for (size_t i = trainCount; i < total; i++) {
				testX.push_back(move(x[i]));
				testY.push_back(move(y[i]));
			}
			x.clear();
			y.clear();

			return { { move(trainX), move(trainY) }, { move(testX), move(testY) } };
		}

		std::pair<DataRow, double> parseLibSVMLine(const std::string & line, size_t featureCount) {
			vector<double> values(featureCount);
			size_t label, index;
			double value;
			int lastp = -1;
			for (size_t p = 0; p < line.length(); p++) {
				if (isspace(line[p]) || p == line.length() - 1) {
					if (lastp == -1) {
						sscanf(line.c_str(), "%zu", &label);
					}
					else {
						sscanf(line.c_str() + lastp, "%zu:%lf", &index, &value);
						if (index > featureCount || index < 1) {
							throw runtime_error("feature index exceeded given dimension " + to_string(featureCount));
						}
						values[index - 1] = value;
					}
					lastp = int(p + 1);
				}
			}
			return { move(values), double(label) };
		}
	}

	long long getCurrentMillisecond() {
		using namespace chrono;
		chrono::milliseconds ms = chrono::duration_cast< chrono::milliseconds >(
			chrono::system_clock::now().time_since_epoch()
			);
		return ms.count();
	}

	std::string readFile(const std::string & path) {
		ifstream file(path);
		if (file.is_open()) {
			string content;
			getline(file, content, '\0');
			return content;
		}
		else {
			throw runtime_error("cannot open " + path);
		}
	}

	double calculateAccuracy(const Data::DataColumn & predict, const Data::DataColumn & truth) {
		if (predict.size() != truth.size()) throw invalid_argument("size not matched");
		int correct = 0;
		for (size_t i = 0; i < predict.size(); i++) {
			if (predict[i] <= .5 && truth[i] == 0.) correct++;
			if (predict[i] > .5 && truth[i] == 1.) correct++;
		}
		return double(correct) / predict.size();
	}

	double calculateAUC(const Data::DataColumn & predict, const Data::DataColumn & truth) {
		if (predict.size() != truth.size()) throw invalid_argument("size not matched");

		vector<pair<double, double>> d(predict.size(), { .0, .0 });
		for (size_t i = 0; i < d.size(); i++) {
			d[i].first = predict[i];
			d[i].second = truth[i];
		}
		sort(d.begin(), d.end(), [](const auto &a, const auto &b) {
			return a.first > b.first;
		});

		size_t nPos = 0, nNeg = 0;
		for (auto v : truth) {
			if (v == 1.) nPos++;
			else nNeg++;
		}

		double auc = .0;
		size_t truePos = 0, falsePos = 0;
		for (const auto &p : d) {
			if (p.second == 1.) truePos++;
			else {
				falsePos++;
				auc += double(truePos);
			}
		}

		return auc / (nPos * nNeg);
	}

	Data::DataColumn & operator-=(Data::DataColumn & a, const Data::DataColumn & b) {
		if (a.size() != b.size()) throw invalid_argument("length not matched");
		for (size_t i = 0; i < a.size(); i++) a[i] -= b[i];
		return a;
	}

	Data::DataColumn & operator+=(Data::DataColumn & a, const Data::DataColumn & b) {
		if (a.size() != b.size()) throw invalid_argument("length not matched");
		for (size_t i = 0; i < a.size(); i++) a[i] += b[i];
		return a;
	}

	Data::DataColumn & operator*=(Data::DataColumn & a, double val) {
		for (auto &v : a) v *= val;
		return a;
	}
}
