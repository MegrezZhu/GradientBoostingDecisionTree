#pragma once

#include <string>
#include <utility>
#include <vector>

namespace zyuco {
	namespace Data {
		typedef std::vector<double> DataRow;
		typedef std::vector<DataRow> DataFrame;
		typedef std::vector<double> DataColumn;
		struct TrainData {
			DataFrame x;
			DataColumn y;
		};
		struct SplitXYResult {
			TrainData train, test;
		};

		TrainData fromLibSVM(const std::string &path, int featureCount);
		SplitXYResult splitXY(DataFrame &&x, DataColumn &&y, double trainSize = 0.8);
		std::pair<DataRow, double> parseLibSVMLine(const std::string &line, int featureCount);
	}

	long long getCurrentMillisecond();
	double calculateAccuracy(const Data::DataColumn &predict, const Data::DataColumn &truth);
	Data::DataColumn& operator-=(Data::DataColumn &a, const Data::DataColumn &b);
	Data::DataColumn& operator+=(Data::DataColumn &a, const Data::DataColumn &b);
	Data::DataColumn& operator*=(Data::DataColumn &a, double val);
}
