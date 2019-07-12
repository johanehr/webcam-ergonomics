#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include "json.hpp"

using namespace nlohmann;
using namespace cv;
int main(int argc, char** argv )
{

    // read a JSON file
    std::ifstream f("config/settings.json");
    json settings;
    f >> settings;

    double ipd = settings["ipd"];
    double alert_time = settings["alert_time"];

    // Print current settings
    std::cout << "Interpupillary distance: " << ipd*1000 << " mm\n";
    std::cout << "Neutral position: (" << settings["neutral"][0] << ", " << settings["neutral"][1] << ", " << settings["neutral"][2] << ") m\n";
    std::cout << "Time before alerted: " << alert_time << " s\n";

    return 0;
}
