#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <chrono>
#include <map>
#include <omp.h>

#include "lib/util.h"
#include "lib/GBDT.h"
#include "lib/logger.h"

struct TaskConfig: zyuco::BoostingConfig {
	size_t maxThreads = 1;
	size_t features;
	double validateSize = .2;
	std::string trainFile;
	std::string testFile;
	std::string predictFile;
};

void handleIO(const std::vector<std::string> &inputs);
void printUsage();
TaskConfig parseConfig(const std::string &str);
std::pair<std::string, std::string> parseConfigItem(const std::string &line);
std::string trim(const std::string &str);
void work(const TaskConfig &config);

using namespace std;
using namespace zyuco;

int main(int argc, char** argv) {
	try {
		vector<string> inputs;
		for (int i = 1; i < argc; i++) {
			inputs.push_back(argv[i]);
		}
		handleIO(inputs);
	}
	catch (const exception &s) {
		cerr << s.what() << endl;
		exit(1);
	}
	catch (...) {
		cerr << "unknown error" << endl;
		exit(1);
	}
}

void handleIO(const std::vector<std::string>& inputs) {
	switch (inputs.size()) {
	case 0:
		printUsage();
		break;
	case 4: {
		auto config = parseConfig(readFile(inputs[0]));
		config.trainFile = inputs[1];
		config.testFile = inputs[2];
		config.predictFile = inputs[3];
		work(config);
		break;
	}
	default:
		cerr << "invalid arguments" << endl;
		printUsage();
	}
}

void printUsage() {
	cout << "usage: boost <config_file> <train_file> <test_file> <predict_dest>" << endl;
}

TaskConfig parseConfig(const std::string &str) {
	const static vector<string> allowedFields = { "eta", "gamma", "maxDepth", "minChildWeight", "rounds", "subsample", "colsampleByTree", "maxThreads", "features", "validateSize" };
	const static vector<string> requiredFileds = { "rounds", "features" };

	map<string, double> configMap;
	stringstream ss(str);
	string rawLine;
	while (getline(ss, rawLine)) {
		auto line = trim(rawLine);
		if (line.empty() || line[0] == '#') continue;
		auto p = parseConfigItem(line);
		if (find(allowedFields.cbegin(), allowedFields.cend(), p.first) != allowedFields.end()) {
			try {
				configMap[p.first] = stod(p.second);
			}
			catch (...) {
				throw runtime_error("invalid value for field '" + p.first + "': '" + p.second + "'");
			}
		}
		else {
			throw runtime_error("unknown config field '" + p.first + "'");
		}
	}

	for (const auto &field : requiredFileds) {
		if (configMap.find(field) == configMap.end()) {
			throw runtime_error("missing required field '" + field + "'");
		}
	}

	TaskConfig config;

	// required fields
	config.rounds = size_t(configMap["rounds"]);
	config.features = size_t(configMap["features"]);

	map<string, double>::const_iterator it;
	// optional fields
#define get_optional_field(name)		\
	it = configMap.find(#name);		\
	config.name = it != configMap.end() ? it->second : config.name
#define get_optional_size_t_field(name) \
	it = configMap.find(#name);		\
	config.name = size_t(it != configMap.end() ? it->second : config.name)

	get_optional_field(eta);
	get_optional_field(gamma);
	get_optional_size_t_field(maxDepth);
	get_optional_size_t_field(minChildWeight);
	get_optional_size_t_field(rounds);
	get_optional_field(subsample);
	get_optional_field(colsampleByTree);
	get_optional_size_t_field(maxThreads);
	get_optional_field(validateSize);

#undef get_optional_field
#undef get_optional_size_t_field

	return config;
}

string trim(const std::string &str) {
	if (str.empty()) return "";
	size_t i = 0, j = str.size() - 1;
	while (i < j && isspace(str[i])) i++;
	while (j > i && isspace(str[j])) j--;
	return str.substr(i, j - i + 1);
}

std::pair<std::string, std::string> parseConfigItem(const std::string &line) {
	auto ind = line.find_first_of('=');
	if (ind == string::npos) throw runtime_error("invalid format '" + line + "'");
	auto first = trim(line.substr(0, ind));
	auto second = trim(line.substr(ind + 1));
	return { move(first), move(second) };
}

void work(const TaskConfig &config) {
#ifdef _OPENMP
	omp_set_num_threads(int(config.maxThreads));
#endif // _OPENMP

	cout << NOW << "reading..." << endl;
	auto data = Data::fromLibSVM(config.trainFile, config.features);
	cout << NOW << "done." << endl;

	unique_ptr<GradientBoostingClassifier> model;
	if (config.validateSize > 0.) {
		auto split = Data::splitXY(move(data.x), move(data.y), 1. - config.validateSize);
		model = GradientBoostingClassifier::fit(split.train.x, split.train.y, config, split.test.x, split.test.y);
		split.train.x.clear();
		split.train.y.clear();
		split.test.x.clear();
		split.test.y.clear();
	}
	else {
		model = GradientBoostingClassifier::fit(data.x, data.y, config);
		data.x.clear();
		data.y.clear();
	}

	// training done, start predicting
	cout << NOW << "start reading data for prediction..." << endl;
	auto testData = Data::fromLibSVM(config.testFile, config.features);
	cout << NOW << "start predicting..." << endl;
	auto testPred = model->predict(testData.x);

	vector<size_t> index(testData.x.size()); // reorder according to test id
	for (size_t i = 0; i < index.size(); i++) index[i] = i;
	sort(index.begin(), index.end(), [&](auto a, auto b) {
		return testData.y[a] < testData.y[b];
	});
	cout << NOW << "done, writing to file..." << endl;

	Data::toCSV(testData.y, testPred, index, config.predictFile);
	cout << NOW << "all done! prediction results are written into " << config.predictFile << endl;
}
