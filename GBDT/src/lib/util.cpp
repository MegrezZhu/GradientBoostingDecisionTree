#include <fstream>
#include <algorithm>
#include <cctype>
#include <sstream>
#include <chrono>
#include <iostream>
#include <omp.h>

#include "util.h"
#include "logger.h"

using namespace std;

namespace zyuco {
	namespace Data {
		TrainData fromLibSVM(const std::string & path, int featureCount) {
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
					x.push_back(item.first);
					y.push_back(item.second);
				}
			}

			return { move(x), move(y) };
		}

		SplitXYResult splitXY(DataFrame &&x, DataColumn &&y, double trainSize) {
			size_t total = x.size();
			size_t trainCount = size_t(total * trainSize), testCount = total - trainCount;

			// shuffle
			vector<size_t> indices(total);
			for (size_t i = 0; i < total; i++) indices[i] = i;
			random_shuffle(indices.begin(), indices.end());

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

			return { { trainX, trainY }, { testX, testY } };
		}

		std::pair<DataRow, double> parseLibSVMLine(const std::string & line, int featureCount) {
			vector<double> values(featureCount);
			size_t label, index;
			double value;
			int lastp = -1;
			for (size_t p = 0; p < line.length(); p++) {
				if (isspace(line[p])) {
					if (lastp == -1) {
						sscanf(line.c_str(), "%llu", &label);
					}
					else {
						sscanf(line.c_str() + lastp, "%llu:%lf", &index, &value);
						values[index] = value;
					}
					lastp = p + 1;
				}
			}
			return { move(values), label };
		}
	}

	long long getCurrentMillisecond() {
		using namespace chrono;
		chrono::milliseconds ms = chrono::duration_cast< chrono::milliseconds >(
			chrono::system_clock::now().time_since_epoch()
			);
		return ms.count();
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
