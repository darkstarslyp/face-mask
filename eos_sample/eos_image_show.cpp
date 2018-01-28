#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "opencv2/opencv.hpp"
#include "ShaderLoader.h"
#include "ldmarkmodel.h"
#include <sys/time.h>
#include "Eos.h"


const GLuint WIDTH = 1080, HEIGHT = 720;

// The MAIN function, from here we start the application and run the game loop
int main()
{
    //初始化Eos
    Eos m_eos;
    if(!m_eos.init()){
        std::cout<<"EOS init failed!!!"<<std::endl;
        return 0;
    }

    //初始化人脸识别
    ldmarkmodel modelt;
    cv::Mat current_shape;
    std::string modelFilePath = "roboman-landmark-model.bin";
    load_ldmarkmodel(modelFilePath, modelt);

    // Paths for shader files
    std::string vtxShader = "../Shaders/defaultVertex.glsl";
    std::string fragShader = "../Shaders/defaultFrag.glsl";

    // Init GLFW
    glfwInit();
    // Set all the required options for GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    // Create a GLFWwindow object that we can use for GLFW's functions
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "OPENGL-FACE-MASK",nullptr, nullptr);
    glfwMakeContextCurrent(window);

    // Options
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    // Set this to true so GLEW knows to use a modern approach to retrieving function pointers and extensions
    glewExperimental = GL_TRUE;
    // Initialize GLEW to setup the OpenGL Function pointers
    glewInit();

    // Build and compile our shader program
    Shader ourShader(vtxShader.c_str(), fragShader.c_str());



    // Generate Glass Model
    std::vector<float> f_vertices = {
            -1.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 0.0f,
            -1.0f, -1.0f, 0.0f,
            -1.0f, -1.0f, 0.0f,
            1.0f, 1.0f, 0.0f,
            1.0f, -1.0f, 0.0f,
    };


    std::vector<float> uv_coords = { 0.0f, 0.0f,
                                     1.0f, 0.0f,
                                     0.0f, 1.0f,
                                     0.0f, 1.0f,
                                     1.0f, 0.0f,
                                     1.0f, 1.0f,
    };


    // ***************
    // Create texture
    // ***************
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    // Set our texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// Set texture wrapping to GL_REPEAT
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // Set texture filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    cv::VideoCapture capture(0);
    capture.set(CV_CAP_PROP_FRAME_WIDTH, WIDTH);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, HEIGHT);


    GLuint VBO, VAO, TBO;

    Mat mask_image(1, 1, CV_8UC4);
    mask_image.at<cv::Vec4b>(0,0) = cv::Vec4b(255,218,185,100);


    float * base_projection = new float[16]{
           1.0f,0.0f,0.0f,0.0f,
           0.0f,1.0f,0.0f,0.0f,
           0.0f,0.0f,1.0f,0.0f,
           0.0f,0.0f,0.0f,1.0f,
    };

    float * mask_projection = new float[16]{
            2.0f,0.0f,0.0f,0.0f,
            0.0f,2.0f,0.0f,0.0f,
            0.0f,0.0f,2.0f,0.0f,
            0.0f,0.0f,0.0f,1.0f,
    };

    // Game loop
    while (!glfwWindowShouldClose(window))
    {


        cv::Mat frame ;
        if (!capture.read(frame)){

            continue;
        }
        cv::cvtColor(frame,frame,CV_BGR2RGB);
        cv::flip(frame,frame,1);
        modelt.track(frame, current_shape);

        int numLandmarks = current_shape.cols / 2;
        std::vector<cv::Point> landmarks;

        for (int j = 0; j < numLandmarks; j++) {

            int x = current_shape.at<float>(j);
            int y = current_shape.at<float>(j + numLandmarks);

            landmarks.push_back(cv::Point(x, y));
            std::stringstream ss;
            ss << j;
            if(false){
                cv::putText(frame, ss.str(), cv::Point(x, y), 0.5, 0.5, cv::Scalar(0, 0, 255));
                cv::circle(frame, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
            }
        }

        EosData eosData;
        int nCoordinates;
        GLfloat* vertices;
        int nUVs ;
        GLfloat* uvs;

        cv::Mat showImage;
        frame.copyTo(showImage);

        if(landmarks.size()==68){
            eosData = m_eos.getMorphModel(frame,landmarks);


            int f_vertices_size = f_vertices.size();
            int eos_vertices_size = eosData.vertices.size();
            int uv_coords_size = uv_coords.size();
            int eos_coords_size = eosData.texcoords.size();

            // Initializing vertex buffer array
            nCoordinates = f_vertices_size+eos_vertices_size;
            vertices = new GLfloat[nCoordinates];

            // Copy data here
            for(int i = 0; i <f_vertices_size; ++i) {
                vertices[i] = f_vertices[i];
            }

            for(int i=0;i<eos_vertices_size;i++){
                vertices[i+f_vertices_size] = eosData.vertices.at(i);
            }

            // Copy uv coordinates to GLfloat[]
            nUVs = uv_coords_size+eos_coords_size;
            uvs = new GLfloat[nUVs];

            for(int i = 0; i < uv_coords_size; ++i) {
                uvs[i] = GLfloat(uv_coords[i]);
            }

            for(int i = 0; i < eos_coords_size; ++i) {
                uvs[i+uv_coords_size] = GLfloat(eosData.texcoords.at(i));
            }

        }else{

            // Initializing vertex buffer array
            nCoordinates = f_vertices.size();
            vertices = new GLfloat[nCoordinates];
            // Copy data here
            for(int i = 0; i <nCoordinates; ++i) {
                vertices[i] = f_vertices[i];
            }

            // Copy uv coordinates to GLfloat[]
            nUVs = uv_coords.size();
            uvs = new GLfloat[nUVs];
            for(int i = 0; i < nUVs; ++i) {
                uvs[i] = GLfloat(uv_coords[i]);
            }
        }

        // ************************************
        // Set up vertx array and vertex buffer
        // ************************************
        glGenVertexArrays(1, &VAO);
        // Vertex Buffer
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &TBO);

        // Bind the Vertex Array Object first, then bind and set vertex buffer(s) and attribute pointer(s).
        // Set up the Model
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, nCoordinates*sizeof(GLfloat), vertices, GL_STATIC_DRAW);

        // Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);
        glEnableVertexAttribArray(0);

        // Texture Buffer
        glBindBuffer(GL_ARRAY_BUFFER, TBO);
        glBufferData(GL_ARRAY_BUFFER, nUVs*sizeof(GLfloat), uvs, GL_STATIC_DRAW);

        // Position attribute
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);
        glEnableVertexAttribArray(1);

        // Bind texture buffer to VAO

        glBindVertexArray(0); // Unbind VAO


        // Check if any events have been activiated (key pressed, mouse moved etc.) and call corresponding response functions
        glfwPollEvents();

        // Render
        // Clear the colorbuffer
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Draw the triangle
        ourShader.Use();
        // Create Camera transformation

        glUniformMatrix4fv(glGetUniformLocation(ourShader.Program, "projection"), 1, GL_FALSE, base_projection);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, showImage.cols, showImage.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, showImage.ptr(0));
        glGenerateMipmap(GL_TEXTURE_2D);
        glUniform1i(glGetUniformLocation(ourShader.Program, "uTexture"), 0);

        glBindVertexArray(VAO);
        // every vertex has 3 coordinates
        glDrawArrays(GL_TRIANGLES, 0, f_vertices.size()/3);
        glBindVertexArray(0);


        if(landmarks.size()==68){

            glUniformMatrix4fv(glGetUniformLocation(ourShader.Program, "projection"), 1, GL_FALSE, base_projection);

            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_LESS);

            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, mask_image.cols, mask_image.rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, mask_image.ptr(0));
            glGenerateMipmap(GL_TEXTURE_2D);
            glUniform1i(glGetUniformLocation(ourShader.Program, "uTexture"), 1);

            glBindVertexArray(VAO);
            // every vertex has 3 coordinates
            glDrawArrays(GL_TRIANGLES, f_vertices.size()/3 ,nCoordinates/ 3);
            glBindVertexArray(0);

            glDisable(GL_BLEND);
            glDisable(GL_DEPTH_TEST);
        }


        // Swap the screen buffers
        glfwSwapBuffers(window);
    }
    // Properly de-allocate all resources once they've outlived their purpose
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    // Terminate GLFW, clearing any resources allocated by GLFW.
    glfwTerminate();

    delete []base_projection;
    delete []mask_projection;
    return 0;
}

