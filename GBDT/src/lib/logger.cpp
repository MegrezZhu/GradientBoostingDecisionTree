#define _CRT_SECURE_NO_WARNINGS

#include "logger.h"

using namespace std;
using namespace chrono;

std::string getTime();

namespace zyuco {
	std::chrono::steady_clock::time_point startPoint = std::chrono::steady_clock::now();
	TimeIndicator NOW;

	std::ostream& operator<<(std::ostream& out, TimeIndicator) {
		out << getTime() << " - ";
		return out;
	}
}

std::string getTime() {
	static char buffer[20];
	long long elapse = duration_cast<milliseconds>(steady_clock::now() - zyuco::startPoint).count();
	auto ms = elapse % 1000;
	auto sec = (elapse / 1000) % 60;
	auto min = (elapse / (60 * 1000)) % 60;
	auto hour = (elapse / (3600 * 1000));
	sprintf(buffer, "%02lld:%02lld:%02lld:%03lld", hour, min, sec, ms);
	return buffer;
}
