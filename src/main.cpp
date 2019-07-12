#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include "json.hpp"

using namespace nlohmann;
using namespace cv;

void handleWebcam( int id , bool live_feed);


void handleWebcam( int id, bool live_feed ){
  VideoCapture cap;

  if(!cap.open(id)){
    std::cout << "Error Opening Capture Device" << std::endl; //Use cerr for basic debugging statements
  }

  while(true){
    Mat frame;
    cap >> frame;
    if (live_feed){
      imshow("webcam-ergonomics LIVE FEED", frame);
      if( waitKey(10) == 27 ) break; // stop capturing by pressing ESC
    }
  }
}

int main(int argc, char** argv )
{
  // Allow for enabled/disabled live feed (to see face etc)
  bool live_feed = false;
  if ( argc == 2 ){
    std::string mode = argv[1];
    if (mode == "-L") {
      live_feed = true;
    }
  }

  // read configuration JSON file
  std::ifstream f("config/settings.json");
  json settings;
  f >> settings;

  double ipd = settings["ipd"];
  double alert_time = settings["alert_time"];

  // Print current settings
  std::cout << "Interpupillary distance: " << ipd*1000 << " mm\n";
  std::cout << "Neutral position: (" << settings["neutral"][0] << ", " << settings["neutral"][1] << ", " << settings["neutral"][2] << ") m\n";
  std::cout << "Time before alerted: " << alert_time << " s\n";

  handleWebcam(settings["camera_id"], live_feed);

  return 0;
}
