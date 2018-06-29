#pragma once

#include <string>
#include <utility>
#include <vector>

namespace zyuco {
	namespace Data {
		typedef std::vector<double> DataRow;
		typedef std::vector<DataRow> DataFrame;
		typedef std::vector<double> DataColumn;
		struct LibSVMData {
			DataFrame x;
			DataColumn y;
		};
		struct SplitXYResult {
			LibSVMData train, test;
		};

		LibSVMData fromLibSVM(const std::string &path, int featureCount);
		void toCSV(const DataColumn &ids, const DataColumn &pred, const std::vector<size_t> &index, const std::string &path);
		SplitXYResult splitXY(DataFrame &&x, DataColumn &&y, double trainSize = 0.8);
		std::pair<DataRow, double> parseLibSVMLine(const std::string &line, int featureCount);
	}

	long long getCurrentMillisecond();
	std::string readFile(const std::string &path);
	double calculateAccuracy(const Data::DataColumn &predict, const Data::DataColumn &truth);
	double calculateAUC(const Data::DataColumn &predict, const Data::DataColumn &truth);

	Data::DataColumn& operator-=(Data::DataColumn &a, const Data::DataColumn &b);
	Data::DataColumn& operator+=(Data::DataColumn &a, const Data::DataColumn &b);
	Data::DataColumn& operator*=(Data::DataColumn &a, double val);
}
