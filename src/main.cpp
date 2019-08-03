#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include "json.hpp"

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <chrono>
#include <thread>

using namespace nlohmann;
using namespace cv;

String path_to_face_cascade;
String path_to_eyes_cascade;

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

Point eye1_center( 0, 0 );
Point eye2_center( 0, 0 );

void handleWebcam( int id , bool live_feed, int downscale);
bool detectEyes( Mat frame, int downscale);

void handleWebcam( int id, bool live_feed, int downscale ){
  VideoCapture cap;
  if(!cap.open(id)){
    std::cout << "Error Opening Capture Device" << std::endl; //Use cerr for basic debugging statements
  }

  if( !face_cascade.load( path_to_face_cascade ) )
  {
      std::cout << "--(!)Error loading face cascade\n";
      return;
  };
  if( !eyes_cascade.load( path_to_eyes_cascade ) )
  {
      std::cout << "--(!)Error loading eyes cascade\n";
      return;
  };

  while(true){

    auto frame_timestamp = std::chrono::system_clock::now();
    Mat frame;
    cap >> frame;

    Mat resized_frame (cvRound(frame.rows / downscale), cvRound( frame.cols / downscale), frame.type());
    resize(frame, resized_frame, resized_frame.size());

    /*
      int im_width = frame.rows;
      int im_height = frame.cols;
      std::cout << "Webcam resolution: " << im_height << "x" << im_width << '\n';
    */
    bool detected = detectEyes( resized_frame, downscale );

    if( waitKey(10) == 27 ) break; // stop capturing by pressing ESC

    //std::this_thread::sleep_until(frame_timestamp + std::chrono::milliseconds(250));
    if (live_feed){
      //ellipse( frame, center, Size( faces[0].width/2, faces[0].height/2 ), 0, 0, 360, Scalar( 255, 0, 0 ), 1 );
      int radius = 40;

      // Mark in white if live, red if old detection
      if (detected){
        circle( frame, eye1_center, radius, Scalar( 255, 255, 255 ), 1 );
        circle( frame, eye2_center, radius, Scalar( 255, 255, 255 ), 1 );
      }
      else {
        circle( frame, eye1_center, radius, Scalar( 0, 0, 255 ), 1 );
        circle( frame, eye2_center, radius, Scalar( 0, 0, 255 ), 1 );
      }

      imshow( "webcam-ergonomics LIVE FEED", frame );
    }
  }
}

bool detectEyes( Mat frame, int downscale)
{
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    //-- Detect faces
    std::vector<Rect> faces;
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 3);

    if (faces.size() == 1) {
      Mat faceROI = frame_gray( faces[0]);

      //-- In each face, detect eyes
      std::vector<Rect> eyes;
      eyes_cascade.detectMultiScale( faceROI, eyes );

      if (eyes.size() == 2){
        // Only updates if finds exactly 2 eyes in the face
        eye1_center.x = (faces[0].x + eyes[0].x + eyes[0].width/2)*downscale;
        eye1_center.y = (faces[0].y + eyes[0].y + eyes[0].height/2)*downscale;
        eye2_center.x = (faces[0].x + eyes[1].x + eyes[1].width/2)*downscale;
        eye2_center.y = (faces[0].y + eyes[1].y + eyes[1].height/2)*downscale;

        std::cout << "Detected both eyes!";
        return true;
      }
    }
    std::cout << "WARNING: Lost sight of eyes!";
    return false;
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
  path_to_eyes_cascade = settings["path_eyes_cascade"];
  path_to_face_cascade = settings["path_face_cascade"];

  // Print current settings
  std::cout << "Interpupillary distance: " << ipd*1000 << " mm\n";
  std::cout << "Neutral position: (" << settings["neutral"][0] << ", " << settings["neutral"][1] << ", " << settings["neutral"][2] << ") m\n";
  std::cout << "Time before alerted: " << alert_time << " s\n";

  handleWebcam(settings["camera_id"], live_feed, settings["downscale_factor"]);

  return 0;
}
