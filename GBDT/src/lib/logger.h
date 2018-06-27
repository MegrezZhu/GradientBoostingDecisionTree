#pragma once

#include <iostream>
#include <string>
#include <chrono>

namespace zyuco {
	struct TimeIndicator {};

	std::ostream& operator<<(std::ostream& out, TimeIndicator);

	extern std::chrono::steady_clock::time_point startPoint;
	extern TimeIndicator NOW;
}
