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

#define TRAILING_AVG_LOCATIONS 10

using namespace nlohmann;
using namespace cv;

class ErgonomicsChecker {
  private:
    int state = 0;
    double alert_time;
    double trailing_position[TRAILING_AVG_LOCATIONS][3];
    double filtered_position[3];
    double neutral_position[3];
    double neutral_radius;
    int num_received = 0;

  public:
    ErgonomicsChecker(){
      readJsonSettings("config/settings.json");
    }

    void readJsonSettings(String file_path){
      std::ifstream f(file_path);
      json settings;
      f >> settings;

      alert_time = settings["alert_time"];
      std::cout << "Bad posture alert after: " << alert_time << " s\n";

      neutral_position[0] = settings["neutral_position"][0];
      neutral_position[1] = settings["neutral_position"][1];
      neutral_position[2] = settings["neutral_position"][2];
      neutral_radius = settings["neutral_radius"];
    }


    void addNewLocation(double x, double y, double z){
      // Keep track of how many locations have been received. Use % operator to select index for storing to continuously update.
      num_received++;
      int index = num_received % TRAILING_AVG_LOCATIONS;
      trailing_position[index][0] = x;
      trailing_position[index][1] = y;
      trailing_position[index][2] = z;

    }

    void calcFilteredLocation(){
      double x_filtered = 0;
      double y_filtered = 0;
      double z_filtered = 0;

      for (int i = 0; i < TRAILING_AVG_LOCATIONS; i++){
        x_filtered += trailing_position[i][0];
        y_filtered += trailing_position[i][1];
        z_filtered += trailing_position[i][2];
      }

      filtered_position[0] = x_filtered / TRAILING_AVG_LOCATIONS;
      filtered_position[1] = y_filtered / TRAILING_AVG_LOCATIONS;
      filtered_position[2] = z_filtered / TRAILING_AVG_LOCATIONS;
    }

    /*

    void checkPosition(){

      // Compare current and neutral position, is it inside neutral bbx?
      // If inside, record OK alert_time

      if ((current_time - last_OK_time) > threshold_time) {
        alertUser();
      }
    }


    void alertUser(){
      beep();
    }

    void beep(){
      std::cout << "\a";
    }
    */
};



class LocationDetector {
  private:
    CascadeClassifier face_cascade;
    CascadeClassifier eyes_cascade;
    VideoCapture cap;
    double downscale_factor;
    int webcam_id;
    double ipd;// in meters
    Point eye1_center = Point( 0, 0 );
    Point eye2_center = Point( 0, 0 );
    Point face_center = Point( 0, 0 );
    Mat frame;
    // TODO: Add camera calibration

  public:
    LocationDetector() { // Constructor with parameters
      readJsonSettings("config/settings.json");

      cap.set(CAP_PROP_BUFFERSIZE, 1);
      if(!cap.open(webcam_id)){
        std::cout << "Error Opening Capture Device" << std::endl; //Use cerr for basic debugging statements
      }
    }

    void readJsonSettings(String file_path){
      std::ifstream f(file_path);
      json settings;
      f >> settings;

      webcam_id = settings["camera_id"];

      ipd = settings["ipd"];
      std::cout << "Interpupillary distance: " << ipd*1000 << " mm\n";

      downscale_factor = settings["downscale_factor"];
      std::cout << "Downscale factor: " << downscale_factor << "x\n";

      if( !face_cascade.load( settings["path_face_cascade"] ) )
        {
          std::cout << "Error loading face cascade\n";
        };
      if( !eyes_cascade.load( settings["path_eyes_cascade"] ) )
        {
          std::cout << "Error loading eyes cascade\n";
        };

      // TODO: Camera calibration settings
    }


    bool captureAndProcessImage() {
      cap >> frame;

      Mat resized_frame (cvRound(frame.rows / downscale_factor), cvRound( frame.cols / downscale_factor), frame.type());
      resize(frame, resized_frame, resized_frame.size());

      Mat resized_frame_gray;
      cvtColor( resized_frame, resized_frame_gray, COLOR_BGR2GRAY );
      equalizeHist( resized_frame_gray, resized_frame_gray );

      bool detected = detectFeatures(resized_frame_gray);
      return detected;
    }

    bool detectFeatures(Mat frame_gray) {
      //-- Detect faces
      std::vector<Rect> faces;
      face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0, Size(30, 30));

      if (faces.size() == 1) {
        face_center.x = (faces[0].x + faces[0].width/2)*downscale_factor;
        face_center.y = (faces[0].y + faces[0].height/2)*downscale_factor;
        Mat faceROI = frame_gray( faces[0] );

        //-- In each face, detect eyes
        std::vector<Rect> eyes;
        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0, Size(6, 6) );

        if (eyes.size() == 2){
          // Only updates if finds exactly 2 eyes in the face
          eye1_center.x = (faces[0].x + eyes[0].x + eyes[0].width/2)*downscale_factor;
          eye1_center.y = (faces[0].y + eyes[0].y + eyes[0].height/2)*downscale_factor;
          eye2_center.x = (faces[0].x + eyes[1].x + eyes[1].width/2)*downscale_factor;
          eye2_center.y = (faces[0].y + eyes[1].y + eyes[1].height/2)*downscale_factor;

          std::cout << "Detected both eyes!\n";
          return true;
        }
        else{
          std::cout << "eyes.size() == " << eyes.size() << "\n";
        }
      }
      else{
        std::cout << "faces.size() == " << faces.size() << "\n";
      }
      std::cout << "WARNING: Lost sight of eyes!\n";
      return false;

    }

    void showLiveFeed(bool detected){
      int radius_eye = frame.cols/20;
      int radius_face = frame.cols/4;

      // Mark in white if newly found, red if old detection
      if (detected){
        circle( frame, face_center, radius_face, Scalar( 255, 255, 255 ), 2 );
        circle( frame, eye1_center, radius_eye, Scalar( 255, 255, 255 ), 1 );
        circle( frame, eye2_center, radius_eye, Scalar( 255, 255, 255 ), 1 );
      }
      else {
        circle( frame, face_center, radius_face, Scalar( 0, 0, 255 ), 2 );
        circle( frame, eye1_center, radius_eye, Scalar( 0, 0, 255 ), 1 );
        circle( frame, eye2_center, radius_eye, Scalar( 0, 0, 255 ), 1 );
      }

      // TODO: Add location (x,y,z) in meters??? Need to be move function call.

      imshow( "webcam-ergonomics LIVE FEED", frame );
    }

    /*
    void calculateLocation(){
      // TODO: return unfiltered location estimate
    }
    */
};


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

  LocationDetector locDet = LocationDetector();
  ErgonomicsChecker ergCheck = ErgonomicsChecker();

  while(true){

    auto t_start = std::chrono::high_resolution_clock::now();
    bool detection_success = locDet.captureAndProcessImage();
    if (detection_success){
      // TODO: Implement logic
      //currLoc = locDet.calculateLocation();
      // ergCheck.addNewLocation(locDet.currLoc);
      // ergCheck.calcFilteredLocation();
      // ergCheck.checkPosition();
    }

    if (live_feed){
      locDet.showLiveFeed(detection_success);
      //locDet.showLiveFeed(detection_success, x, y, z, time_NOT_OK);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsedTime = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    std::cout << "One loop took: "<<elapsedTime<<" ms.\n";

    if( waitKey(10) == 27 ) break; // stop upon pressing ESC key
  }
  return 0;
}
