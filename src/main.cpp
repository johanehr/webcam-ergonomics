#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include "json.hpp"

#include <opencv2/opencv.hpp>
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
    std::chrono::time_point<std::chrono::high_resolution_clock> last_OK_time = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> last_alert = std::chrono::high_resolution_clock::now();

  public:
    ErgonomicsChecker(){
      readJsonSettings("config/settings.json");
    }

    double getAlertTime(){
      return alert_time;
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> getLastOKTime(){
      return last_OK_time;
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
      int index = num_received % TRAILING_AVG_LOCATIONS;
      trailing_position[index][0] = x;
      trailing_position[index][1] = y;
      trailing_position[index][2] = z;
      num_received++;
    }

    void calcFilteredLocation(){
      double x_filtered = 0;
      double y_filtered = 0;
      double z_filtered = 0;

      for (int i = 0; i < TRAILING_AVG_LOCATIONS; i++){
        // NOTE: Before everything is filled, what data is in matrix? Just garbage values?
        x_filtered += trailing_position[i][0];
        y_filtered += trailing_position[i][1];
        z_filtered += trailing_position[i][2];
      }

      filtered_position[0] = x_filtered / TRAILING_AVG_LOCATIONS;
      filtered_position[1] = y_filtered / TRAILING_AVG_LOCATIONS;
      filtered_position[2] = z_filtered / TRAILING_AVG_LOCATIONS;
    }

    bool checkErgonomics(){

      // Compare current and neutral position, is it sufficiently close to neutral position?
      double distance = sqrt(
        pow(filtered_position[0]-neutral_position[0],2)
        +pow(filtered_position[1]-neutral_position[1],2)
        +pow(filtered_position[2]-neutral_position[2],2));

      bool good_posture = false;
      // If inside, record OK alert_time
      if (distance <= neutral_radius){
        last_OK_time = std::chrono::high_resolution_clock::now();
        good_posture = true;
      }

      // If time since OK exceeds alert_time, alert the user
      double time_since_OK = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - last_OK_time).count()/1000.0;
      alertUser(alert_time - time_since_OK);

      return good_posture;
    }

    // countdown is in seconds. Values less than zero warrant an auditory alert
    void alertUser(double countdown){

      if (std::abs(countdown) < 0.05){
        // Initial beep
        std::cout << "\a";
        last_alert = std::chrono::high_resolution_clock::now();
      }

      else if (countdown < 0){

        double time_since_last_alert = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - last_alert).count()/1000.0;
        double beep_period;

        if (countdown < -60.0){
          // Bad posture for a long time (60+s), beep every 1 second
          beep_period = 1.0;

        }
        else if (countdown < -30.0){
          // Bad posture for a medium time (30-60s), beep every 5 s
          beep_period = 5.0;
        }
        else{
          // Bad posture for a short amount of time (0-30s), beep every 10 s
          beep_period = 10.0;
        }

        if (time_since_last_alert > beep_period){
          std::cout << "\a";
          last_alert = std::chrono::high_resolution_clock::now();
        }


      }
    }

};



class LocationDetector {
  private:
    CascadeClassifier face_cascade;
    CascadeClassifier eyes_cascade;
    VideoCapture cap;
    double downscale_factor;
    int webcam_id;
    double ipd;// in meters
    double focal_length;
    Point eye1_center = Point( 0, 0 );
    Point eye2_center = Point( 0, 0 );
    Point face_center = Point( 0, 0 );
    Mat frame;
    bool showResolutionOnce = false; // used to only show webcam resolution once


  public:
    LocationDetector() { // Constructor with parameters
      readJsonSettings("config/settings.json");

      cap.set(CAP_PROP_BUFFERSIZE, 1);
      if(!cap.open(webcam_id)){
        std::cout << "Error Opening Capture Device" << std::endl; //Use cerr for basic debugging statements
      }
    }

    // Coordinates relative to camera, z is depth
    double xCoord;
    double yCoord;
    double zCoord;

    void readJsonSettings(String file_path){
      std::ifstream f(file_path);
      json settings;
      f >> settings;

      webcam_id = settings["camera_id"];

      ipd = settings["ipd"];
      std::cout << "Interpupillary distance: " << ipd*1000 << " mm\n";

      downscale_factor = settings["downscale_factor"];
      std::cout << "Downscale factor: " << downscale_factor << "x\n";

      focal_length = settings["camera_calibration"]["f"];
      std::cout << "Focal length: " << focal_length << "\n";

      if( !face_cascade.load( settings["path_face_cascade"] ) )
        {
          std::cout << "Error loading face cascade\n";
        };
      if( !eyes_cascade.load( settings["path_eyes_cascade"] ) )
        {
          std::cout << "Error loading eyes cascade\n";
        };

    }


    int captureAndProcessImage() {
      cap >> frame;

      if (!showResolutionOnce){
        std::cout << "Webcam resolution: " << frame.cols << "x" << frame.rows << " px\n";
        showResolutionOnce = true;
      }

      Mat resized_frame (cvRound(frame.rows / downscale_factor), cvRound( frame.cols / downscale_factor), frame.type());
      resize(frame, resized_frame, resized_frame.size());

      Mat resized_frame_gray;
      cvtColor( resized_frame, resized_frame_gray, COLOR_BGR2GRAY );
      equalizeHist( resized_frame_gray, resized_frame_gray );

      int detected_state = detectFeatures(resized_frame_gray);

      return detected_state;
    }

    int detectFeatures(Mat frame_gray) {
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

          return 2; // Found face with 2 eyes
        }
        else{
          return 1; // Found face, but not eyes
        }
      }

      // Lost sight of face completely
      return 0;

    }

    void showLiveFeed(int detection_state, double countdown){

      int radius_eye = frame.cols/20;
      int radius_face = frame.cols/4;

      Scalar white = Scalar( 255, 255, 255 );
      Scalar orange = Scalar( 0, 128, 255 );
      Scalar red = Scalar( 0, 0, 255 );

      // Mark in white if newly found, red if old detection
      if (detection_state == 2){ // Found both face and eyes
        circle( frame, face_center, radius_face, white, 2 );
        circle( frame, eye1_center, radius_eye, white, 1 );
        circle( frame, eye2_center, radius_eye, white, 1 );
      }
      else if (detection_state == 1){ // Found only face
        circle( frame, face_center, radius_face, white, 2 );
        circle( frame, eye1_center, radius_eye, red, 1 );
        circle( frame, eye2_center, radius_eye, red, 1 );
      }
      else { // Did not find at all
        circle( frame, face_center, radius_face, red, 2 );
        circle( frame, eye1_center, radius_eye, red, 1 );
        circle( frame, eye2_center, radius_eye, red, 1 );
      }

      // Format decimals for presentation
      std::stringstream stream;
      stream << std::fixed << std::setprecision(2) << xCoord;
      std::string xCoord_str = stream.str();
      stream.str(std::string());

      stream << std::fixed << std::setprecision(2) << yCoord;
      std::string yCoord_str = stream.str();
      stream.str(std::string());

      stream << std::fixed << std::setprecision(2) << zCoord;
      std::string zCoord_str = stream.str();
      stream.str(std::string());

      stream << std::fixed << std::setprecision(1) << countdown;
      std::string countdown_str = stream.str();
      stream.str(std::string());

      std::string pos_text = "POSITION: (" + xCoord_str + ", " + yCoord_str + ", " + zCoord_str + ")";
      std::string countdown_text = "COUNTDOWN: " + countdown_str + "s";

      Scalar font_color;
      Scalar font_countdown_color;

      if (countdown > 9.0){
        font_color = white;
        font_countdown_color = white;
      }
      else if (countdown > 5.0){
        font_color = red;
        font_countdown_color = white;
      }
      else if (countdown > 0.0){
        font_color = red;
        font_countdown_color = orange;
      }
      else {
        font_color = red;
        font_countdown_color = red;
      }

      putText( frame, pos_text, Point(frame.cols/20, frame.cols/20), FONT_HERSHEY_SIMPLEX, 0.5, font_color );
      putText( frame, countdown_text, Point(frame.cols/20, frame.cols/10), FONT_HERSHEY_SIMPLEX, 0.5, font_countdown_color );

      imshow( "webcam-ergonomics LIVE FEED", frame );
    }

    void calculateLocation(){
      // Use basic projector model with "known" distance to eyes based on known IPD and focal length. Assumption: face looking directly at camera.
      double px_between_eyes = sqrt(pow(eye1_center.x - eye2_center.x, 2.0) + pow(eye1_center.y - eye2_center.y, 2.0));
      zCoord = ipd * focal_length / px_between_eyes;

      double x_avg = (eye1_center.x + eye2_center.x) / 2.0;
      double y_avg = (eye1_center.y + eye2_center.y) / 2.0;

      double w = frame.cols; // frame width in px
      double h = frame.rows; // frame height in px

      xCoord = (x_avg - w/2.0)*zCoord/focal_length;
      yCoord = (y_avg - h/2.0)*zCoord/focal_length;
    }
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
    /*
    TODO: Make a "set neutral" mode, where pressing a key sets the position
    if (mode == "-S") {
      neutral_set = false;
    }
    */
  }

  LocationDetector locDet = LocationDetector();
  ErgonomicsChecker ergCheck = ErgonomicsChecker();

  while(true){

    auto t_start = std::chrono::high_resolution_clock::now();
    int detection_state = locDet.captureAndProcessImage();
    if (detection_state == 2){
      locDet.calculateLocation();
      ergCheck.addNewLocation(locDet.xCoord, locDet.yCoord, locDet.zCoord);
      ergCheck.calcFilteredLocation();
    }

    // Regardless of whether location detected, use latest valid data to check ergo
    bool good_posture = ergCheck.checkErgonomics();

    if (live_feed){

      double seconds_since_OK = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - ergCheck.getLastOKTime()).count()/1000.0;
      locDet.showLiveFeed(detection_state, ergCheck.getAlertTime() - seconds_since_OK);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsedTime = std::chrono::duration<double, std::milli>(t_end-t_start).count();

    std::string posture_text;
    if (good_posture){
      posture_text = "GOOD";
    }
    else {
      posture_text = "POOR";
    }
    std::cout << "\rPosture: " << posture_text << " --- script running at: ~"<<(int)(1000.0/elapsedTime)<< " Hz   " << std::flush;

    if( waitKey(10) == 27 ) break; // stop upon pressing ESC key when preview is in focus

  }
  return 0;
}
