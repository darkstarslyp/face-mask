//
// Created by 李运平 on 2018/1/26.
//

#include "Eos.h"

Eos::Eos() {

}

Eos::~Eos() {

}

/**
 *
 * @return
 */
bool Eos::init() {

    std::string modelfile = "../data/eos/sfm_shape_3448.bin";
    std::string mappingsfile = "../data/eos/ibug_to_sfm.txt";
    std::string contourfile = "../data/eos/sfm_model_contours.json";
    std::string edgetopologyfile = "../data/eos/sfm_3448_edge_topology.json";
    std::string blendshapesfile = "../data/eos/expression_blendshapes_3448.bin";

    ///加载低精度的 3D morphable model
    try {
        m_morphable_model = morphablemodel::load_model(modelfile);
    } catch (const std::runtime_error &e) {
        cout << "Error loading the Morphable Model: " << e.what() << endl;
        return false;
    }

    // The landmark mapper is used to map 2D landmark points (e.g. from the ibug scheme) to vertex ids:
    try {
        m_landmark_mapper = core::LandmarkMapper(mappingsfile);
    } catch (const std::exception &e) {
        cout << "Error loading the landmark mappings: " << e.what() << endl;
        return false;
    }

    // The expression blendshapes:
    m_blendshapes = morphablemodel::load_blendshapes(blendshapesfile);

    m_model_contour = contourfile.empty() ? fitting::ModelContour() : fitting::ModelContour::load(contourfile);

    m_ibug_contour = fitting::ContourLandmarks::load(mappingsfile);

    m_edge_topology = morphablemodel::load_edge_topology(edgetopologyfile);

    m_init_flag = true;
    return m_init_flag;
}

/**
 * 返回人脸的3D模型信息 EosData
 * @param rgbImage
 * @param landmarks
 * @return
 */
EosData Eos::getMorphModel(cv::Mat rgbImage, std::vector<cv::Point> landmarks) {

    if (landmarks.size() != 68) {
        return EosData();
    }

    EosData eosData = EosData();

    LandmarkCollection<Eigen::Vector2f> eos_landmarks;

    int landmark_size = landmarks.size();
    for (int i = 0; i < landmark_size; i++) {
        Landmark<Eigen::Vector2f> eos_point;
        cv::Point point = landmarks.at(i);

        eos_point.name = std::to_string(i + 1);
        eos_point.coordinates[0] = float(point.x);
        eos_point.coordinates[1] = float(point.y);
        eos_landmarks.push_back(eos_point);
    }

    // Fit the model, get back a mesh and the pose:
    core::Mesh mesh;
    fitting::RenderingParameters rendering_params;

    std::tie(mesh, rendering_params) = fitting::fit_shape_and_pose(
            m_morphable_model, m_blendshapes, eos_landmarks, m_landmark_mapper, rgbImage.cols, rgbImage.rows,
            m_edge_topology,
            m_ibug_contour, m_model_contour, 1, std::experimental::nullopt, 30.0f);

    // The 3D head pose can be recovered as follows:
    float yaw_angle = glm::degrees(glm::yaw(rendering_params.get_rotation()));
    // and similarly for pitch and roll.

    // Extract the texture from the rgbImage using given mesh and camera parameters:
    const Eigen::Matrix<float, 3, 4> affine_from_ortho =
            fitting::get_3x4_affine_camera_matrix(rendering_params, rgbImage.cols, rgbImage.rows);


    render::draw_wireframe(eosData.outputImage, mesh, rendering_params.get_modelview(),
                           rendering_params.get_projection(),
                           fitting::get_opencv_viewport(rgbImage.cols, rgbImage.rows));


    for (const auto &triangle : mesh.tvi) {

//        eosData.vertices.push_back(mesh.vertices[triangle[0]][0]);
//        eosData.vertices.push_back(mesh.vertices[triangle[0]][1]);
//        eosData.vertices.push_back(0.0);
//
//        eosData.vertices.push_back(mesh.vertices[triangle[1]][0]);
//        eosData.vertices.push_back(mesh.vertices[triangle[1]][1]);
//        eosData.vertices.push_back(0.0);
//
//        eosData.vertices.push_back(mesh.vertices[triangle[2]][0]);
//        eosData.vertices.push_back(mesh.vertices[triangle[2]][1]);
//        eosData.vertices.push_back(0.0);


        eosData.texcoords.push_back(mesh.texcoords[triangle[0]][0]);
        eosData.texcoords.push_back(mesh.texcoords[triangle[0]][1]);

        eosData.texcoords.push_back(mesh.texcoords[triangle[1]][0]);
        eosData.texcoords.push_back(mesh.texcoords[triangle[1]][1]);

        eosData.texcoords.push_back(mesh.texcoords[triangle[2]][0]);
        eosData.texcoords.push_back(mesh.texcoords[triangle[2]][1]);

    }

    eosData.vertices = getMappedPoint(rgbImage, mesh, rendering_params.get_modelview(),
                                      rendering_params.get_projection(),
                                      fitting::get_opencv_viewport(rgbImage.cols, rgbImage.rows));


    eosData.projection = rendering_params.get_projection() * rendering_params.get_modelview();

    return eosData;
}


std::vector<float> Eos::getMappedPoint(cv::Mat image, const core::Mesh &mesh, glm::mat4x4 modelview,
                                       glm::mat4x4 projection, glm::vec4 viewport) {

    std::vector<float> mapped_point;
    for (const auto &triangle : mesh.tvi) {
        const auto p1 = glm::project(
                {mesh.vertices[triangle[0]][0], mesh.vertices[triangle[0]][1], mesh.vertices[triangle[0]][2]},
                modelview, projection, viewport);
        const auto p2 = glm::project(
                {mesh.vertices[triangle[1]][0], mesh.vertices[triangle[1]][1], mesh.vertices[triangle[1]][2]},
                modelview, projection, viewport);
        const auto p3 = glm::project(
                {mesh.vertices[triangle[2]][0], mesh.vertices[triangle[2]][1], mesh.vertices[triangle[2]][2]},
                modelview, projection, viewport);

        mapped_point.push_back(1 - 2 * (1 - p1.x / image.cols));
        mapped_point.push_back(1 - 2 * p1.y / image.rows);
        mapped_point.push_back(0);

        mapped_point.push_back(1 - 2 * (1 - p2.x / image.cols));
        mapped_point.push_back(1 - 2 * p2.y / image.rows);
        mapped_point.push_back(0);

        mapped_point.push_back(1 - 2 * (1 - p3.x / image.cols));
        mapped_point.push_back(1 - 2 * p3.y / image.rows);
        mapped_point.push_back(0);

    }

    return mapped_point;
}



