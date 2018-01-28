//
// Created by 李运平 on 2018/1/26.
//

#ifndef OPENGLDEMO_EOS_H
#define OPENGLDEMO_EOS_H


#include "eos/core/Image.hpp"
#include "eos/core/Image_opencv_interop.hpp"
#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/core/read_pts_landmarks.hpp"
#include "eos/fitting/fitting.hpp"
#include "eos/render/draw_utils.hpp"
#include "Eigen/Core"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <experimental/optional>
#include <string>
#include <vector>

using namespace eos;
using eos::core::Landmark;
using eos::core::LandmarkCollection;
using cv::Mat;
using std::cout;
using std::endl;
using std::string;
using std::vector;

struct EosData{

    std::vector<float> vertices;
    std::vector<float> texcoords;
    cv::Mat outputImage;
    glm::mat4x4 projection;
};

class Eos {

public:
    Eos();
    ~Eos();
    bool init();
    EosData getMorphModel(cv::Mat rgbImage,std::vector<cv::Point> landmarks);

private:
    std::vector<float > getMappedPoint(cv::Mat image, const core::Mesh& mesh, glm::mat4x4 modelview,
                                       glm::mat4x4 projection, glm::vec4 viewport);

private:
    morphablemodel::MorphableModel m_morphable_model;
    core::LandmarkMapper m_landmark_mapper;
    vector<morphablemodel::Blendshape> m_blendshapes;
    fitting::ModelContour m_model_contour;
    fitting::ContourLandmarks m_ibug_contour;
    morphablemodel::EdgeTopology m_edge_topology;

    bool m_init_flag = false;
};

#endif //OPENGLDEMO_EOS_H
