#include "FaceLandmark.hpp"
#include <iostream>


#define HAND_LANDMARKS 21
/*
Helper function
*/
bool __isIndexValid(int idx) {
    if (idx < 0 || idx >= HAND_LANDMARKS) {
        std::cerr << "Index " << idx << " is out of range (" \
        << HAND_LANDMARKS << ")." << std::endl;
        return false;
    }
    return true;
}


my::HandLandmark::HandLandmark(std::string modelPath):
    HandDetection(modelPath),
    m_landmarkModel(modelPath + std::string("/hand_landmark_full.tflite"))
    {}


void my::HandLandmark::process() {
    HandDetection::process();
    auto roi = HandDetection::getFaceRoi();
    if (roi.empty()) return;

    auto face = HandDetection::cropFrame(roi);
    m_landmarkModel.loadImageToInput(face);
    m_landmarkModel.process();
}


cv::Point my::HandLandmark::getHandLandmarkAt(int index) const {
    if (__isIndexValid(index)) {
        auto roi = HandDetection::getFaceRoi();

        float _x = m_landmarkModel.getOutputData()[index * 3];
        float _y = m_landmarkModel.getOutputData()[index * 3 + 1];

        int x = (int)(_x / m_landmarkModel.getInputShape()[2] * roi.width) + roi.x;
        int y = (int)(_y / m_landmarkModel.getInputShape()[1] * roi.height) + roi.y;

        return cv::Point(x,y);
    }
    return cv::Point();
}


std::vector<cv::Point> my::HandLandmark::getAllHandLandmarks() const {
    if (HandDetection::getFaceRoi().empty())
        return std::vector<cv::Point>();

    std::vector<cv::Point> landmarks(HAND_LANDMARKS);
    for (int i = 0; i < HAND_LANDMARKS; ++i) {
        landmarks[i] = getHandLandmarkAt(i);
    }
    return landmarks;
}


std::vector<float> my::HandLandmark::loadOutput(int index) const {
    return m_landmarkModel.loadOutput();
}