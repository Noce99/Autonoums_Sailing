#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "ImportedModel.cpp"
#include "Utils.h"
#include "SBA_socket.cpp"

using namespace std;

const char  obj_folder []  = "./resources/objs/";
const char  shader_folder []  = "./shaders/";

#define numVAOs 2
#define numVBOs 4
//VAO = Vertex Array Objects
//VBO = Vertex Buffer Objects

#define PASSOVISUALE 0.1f
#define PASSOCAMERA 1.0f
#define PASSOVELA 0.01

float cameraX , cameraY, cameraZ;
float lookingDirX, lookingDirY, lookingDirZ;
float boatLocX, boatLocY, boatLocZ;

bool cameraAttacedToBoat = false;
float backmousex=0, backmousey=0;
bool mouseblocked = false;

glm::vec3 upCamera;
GLuint renderingProgram; //GLuint è una shortcat per unsigned int
GLuint vao[numVAOs];
GLuint vbo_sail[numVBOs];
GLuint vbo_boat[numVBOs];
GLuint vbo_insland[numVBOs];
GLuint vbo_windarrow[numVBOs];
GLuint vbo_currentarrow[numVBOs];
GLuint vbo_forcearrow[numVBOs];
GLuint vbo_rudder[numVBOs];
GLuint vbo_clock_momentum[numVBOs];
GLuint vbo_anticlock_momentum[numVBOs];

//Variabili allocate in init così non devono essere allocate durante il rendering
GLuint mvLoc, projLoc, nLoc;
GLuint globalAmbLoc, ambLoc, diffLoc, specLoc, posLoc, mAmbLoc, mDiffLoc, mSpecLoc, mShiLoc;
int width, height;
float aspect;
glm::mat4 pMat, vMat, mMat, mvMat, invTrMat;
/*
 * pMat : Perspective matrix
 * vMat : View matrix
 * mMat : Model matrix
 * mvMat : vMat*mMat
 * invTrMat : Inverse Trasponse vMat
*/
glm::vec3 currentLightPos, lightPosV; //Light position in model and view space
float lightPos[3]; //Light Position in array di float
glm::vec3 initialLightLoc = glm::vec3(0.0f, 100.0f, 0.0f);// Initial Light position

//WHITE LIGHT PROPRIETY
float globalAmbient[4] = {2.0f, 2.0f, 2.0f, 1.0f};
float lightAmbient[4] = {0.1f, 0.1f, 0.1f, 1.0f};
float lightDiffuse[4] = {1.0f, 1.0f, 1.0f, 1.0f};
float lightSpecular[4] = {1.0f, 1.0f, 1.0f, 1.0f};

//gold material proprieties
float * matAmb = goldAmbient();
float * matDif = goldDiffuse();
float * matSpe = goldSpecular();
float matShi = goldShininess();
void installLights(glm::mat4);

ImportedModel sail(concat(obj_folder, "Sail.obj"));
ImportedModel boat(concat(obj_folder, "Sail_Boat_Without_Sails.obj"));
ImportedModel insland(concat(obj_folder, "Colline.obj"));
ImportedModel windarrow(concat(obj_folder, "Wind_Arrow.obj"));
ImportedModel currentarrow(concat(obj_folder, "Current_Arrow.obj"));
ImportedModel forcearrow(concat(obj_folder, "Force_Arrow.obj"));
ImportedModel rudder(concat(obj_folder, "Rudder.obj"));
ImportedModel clock_momentum(concat(obj_folder, "Clock_Momentum.obj"));
ImportedModel anticlock_momentum(concat(obj_folder, "AntiClock_Momentum.obj"));

std::vector<double> recived_values = std::vector<double>(maxnumofvalues);
double lambda_to_send = 0;
double sigma_to_send = 0;


void setupVertices(void){
    std::vector<glm::vec3> vert_boat = boat.getVertices();
    std::vector<glm::vec2> tex_boat = boat.getTextCoords();
    std::vector<glm::vec3> norm_boat = boat.getNormalVecs();
    int numObjVertices = boat.getNumVertices();

    std::vector<float> pvalues_boat; //vertex positions
    std::vector<float> tvalues_boat; //texture coordinates
    std::vector<float> nvalues_boat; //normal vectors

    for (int i=0; i<numObjVertices; i++){
        pvalues_boat.push_back((vert_boat[i]).x);
        pvalues_boat.push_back((vert_boat[i]).y);
        pvalues_boat.push_back((vert_boat[i]).z);
        tvalues_boat.push_back((tex_boat[i]).s);
        tvalues_boat.push_back((tex_boat[i]).t);
        nvalues_boat.push_back((norm_boat[i]).x);
        nvalues_boat.push_back((norm_boat[i]).y);
        nvalues_boat.push_back((norm_boat[i]).z);
    }
    glGenVertexArrays(1, vao);
    glBindVertexArray(vao[0]);
    glGenBuffers(numVBOs, vbo_boat);

    //VBO for vertex location
    glBindBuffer(GL_ARRAY_BUFFER, vbo_boat[0]);
    glBufferData(GL_ARRAY_BUFFER, pvalues_boat.size()*4, &pvalues_boat[0], GL_STATIC_DRAW);

    //VBO for texture coordinates
    glBindBuffer(GL_ARRAY_BUFFER, vbo_boat[1]);
    glBufferData(GL_ARRAY_BUFFER, tvalues_boat.size()*4, &tvalues_boat[0], GL_STATIC_DRAW);

    //VBO for normal vectors
    glBindBuffer(GL_ARRAY_BUFFER, vbo_boat[2]);
    glBufferData(GL_ARRAY_BUFFER, nvalues_boat.size()*4, &nvalues_boat[0], GL_STATIC_DRAW);

    std::vector<glm::vec3> vert_sail = sail.getVertices();
    std::vector<glm::vec2> tex_sail = sail.getTextCoords();
    std::vector<glm::vec3> norm_sail = sail.getNormalVecs();
    numObjVertices = sail.getNumVertices();

    std::vector<float> pvalues_sail; //vertex positions
    std::vector<float> tvalues_sail; //texture coordinates
    std::vector<float> nvalues_sail; //normal vectors

    for (int i=0; i<numObjVertices; i++){
        pvalues_sail.push_back((vert_sail[i]).x);
        pvalues_sail.push_back((vert_sail[i]).y);
        pvalues_sail.push_back((vert_sail[i]).z);
        tvalues_sail.push_back((tex_sail[i]).s);
        tvalues_sail.push_back((tex_sail[i]).t);
        nvalues_sail.push_back((norm_sail[i]).x);
        nvalues_sail.push_back((norm_sail[i]).y);
        nvalues_sail.push_back((norm_sail[i]).z);
    }
    glGenVertexArrays(1, vao);
    glBindVertexArray(vao[0]);
    glGenBuffers(numVBOs, vbo_sail);

    //VBO for vertex location
    glBindBuffer(GL_ARRAY_BUFFER, vbo_sail[0]);
    glBufferData(GL_ARRAY_BUFFER, pvalues_sail.size()*4, &pvalues_sail[0], GL_STATIC_DRAW);

    //VBO for texture coordinates
    glBindBuffer(GL_ARRAY_BUFFER, vbo_sail[1]);
    glBufferData(GL_ARRAY_BUFFER, tvalues_sail.size()*4, &tvalues_sail[0], GL_STATIC_DRAW);

    //VBO for normal vectors
    glBindBuffer(GL_ARRAY_BUFFER, vbo_sail[2]);
    glBufferData(GL_ARRAY_BUFFER, nvalues_sail.size()*4, &nvalues_sail[0], GL_STATIC_DRAW);

    std::vector<glm::vec3> vert_island = insland.getVertices();
    std::vector<glm::vec2> tex_island = insland.getTextCoords();
    std::vector<glm::vec3> norm_island = insland.getNormalVecs();
    numObjVertices = insland.getNumVertices();

    std::vector<float> pvalues_insland; //vertex positions
    std::vector<float> tvalues_insland; //texture coordinates
    std::vector<float> nvalues_insland; //normal vectors

    for (int i=0; i<numObjVertices; i++){
        pvalues_insland.push_back((vert_island[i]).x);
        pvalues_insland.push_back((vert_island[i]).y);
        pvalues_insland.push_back((vert_island[i]).z);
        tvalues_insland.push_back((tex_island[i]).s);
        tvalues_insland.push_back((tex_island[i]).t);
        nvalues_insland.push_back((norm_island[i]).x);
        nvalues_insland.push_back((norm_island[i]).y);
        nvalues_insland.push_back((norm_island[i]).z);
    }
    glGenVertexArrays(1, vao);
    glBindVertexArray(vao[0]);
    glGenBuffers(numVBOs, vbo_insland);

    //VBO for vertex location
    glBindBuffer(GL_ARRAY_BUFFER, vbo_insland[0]);
    glBufferData(GL_ARRAY_BUFFER, pvalues_insland.size()*4, &pvalues_insland[0], GL_STATIC_DRAW);

    //VBO for texture coordinates
    glBindBuffer(GL_ARRAY_BUFFER, vbo_insland[1]);
    glBufferData(GL_ARRAY_BUFFER, tvalues_insland.size()*4, &tvalues_insland[0], GL_STATIC_DRAW);

    //VBO for normal vectors
    glBindBuffer(GL_ARRAY_BUFFER, vbo_insland[2]);
    glBufferData(GL_ARRAY_BUFFER, nvalues_insland.size()*4, &nvalues_insland[0], GL_STATIC_DRAW);

    std::vector<glm::vec3> vert_windarrow = windarrow.getVertices();
    std::vector<glm::vec2> tex_windarrow = windarrow.getTextCoords();
    std::vector<glm::vec3> norm_windarrow = windarrow.getNormalVecs();
    numObjVertices = windarrow.getNumVertices();

    std::vector<float> pvalues_windarrow; //vertex positions
    std::vector<float> tvalues_windarrow; //texture coordinates
    std::vector<float> nvalues_windarrow; //normal vectors

    for (int i=0; i<numObjVertices; i++){
        pvalues_windarrow.push_back((vert_windarrow[i]).x);
        pvalues_windarrow.push_back((vert_windarrow[i]).y);
        pvalues_windarrow.push_back((vert_windarrow[i]).z);
        tvalues_windarrow.push_back((tex_windarrow[i]).s);
        tvalues_windarrow.push_back((tex_windarrow[i]).t);
        nvalues_windarrow.push_back((norm_windarrow[i]).x);
        nvalues_windarrow.push_back((norm_windarrow[i]).y);
        nvalues_windarrow.push_back((norm_windarrow[i]).z);
    }
    glGenVertexArrays(1, vao);
    glBindVertexArray(vao[0]);
    glGenBuffers(numVBOs, vbo_windarrow);

    //VBO for vertex location
    glBindBuffer(GL_ARRAY_BUFFER, vbo_windarrow[0]);
    glBufferData(GL_ARRAY_BUFFER, pvalues_windarrow.size()*4, &pvalues_windarrow[0], GL_STATIC_DRAW);

    //VBO for texture coordinates
    glBindBuffer(GL_ARRAY_BUFFER, vbo_windarrow[1]);
    glBufferData(GL_ARRAY_BUFFER, tvalues_windarrow.size()*4, &tvalues_windarrow[0], GL_STATIC_DRAW);

    //VBO for normal vectors
    glBindBuffer(GL_ARRAY_BUFFER, vbo_windarrow[2]);
    glBufferData(GL_ARRAY_BUFFER, nvalues_windarrow.size()*4, &nvalues_windarrow[0], GL_STATIC_DRAW);

    std::vector<glm::vec3> vert_currentarrow = currentarrow.getVertices();
    std::vector<glm::vec2> tex_currentarrow = currentarrow.getTextCoords();
    std::vector<glm::vec3> norm_currentarrow = currentarrow.getNormalVecs();
    numObjVertices = currentarrow.getNumVertices();

    std::vector<float> pvalues_currentarrow; //vertex positions
    std::vector<float> tvalues_currentarrow; //texture coordinates
    std::vector<float> nvalues_currentarrow; //normal vectors

    for (int i=0; i<numObjVertices; i++){
        pvalues_currentarrow.push_back((vert_currentarrow[i]).x);
        pvalues_currentarrow.push_back((vert_currentarrow[i]).y);
        pvalues_currentarrow.push_back((vert_currentarrow[i]).z);
        tvalues_currentarrow.push_back((tex_currentarrow[i]).s);
        tvalues_currentarrow.push_back((tex_currentarrow[i]).t);
        nvalues_currentarrow.push_back((norm_currentarrow[i]).x);
        nvalues_currentarrow.push_back((norm_currentarrow[i]).y);
        nvalues_currentarrow.push_back((norm_currentarrow[i]).z);
    }
    glGenVertexArrays(1, vao);
    glBindVertexArray(vao[0]);
    glGenBuffers(numVBOs, vbo_currentarrow);

    //VBO for vertex location
    glBindBuffer(GL_ARRAY_BUFFER, vbo_currentarrow[0]);
    glBufferData(GL_ARRAY_BUFFER, pvalues_currentarrow.size()*4, &pvalues_currentarrow[0], GL_STATIC_DRAW);

    //VBO for texture coordinates
    glBindBuffer(GL_ARRAY_BUFFER, vbo_currentarrow[1]);
    glBufferData(GL_ARRAY_BUFFER, tvalues_currentarrow.size()*4, &tvalues_currentarrow[0], GL_STATIC_DRAW);

    //VBO for normal vectors
    glBindBuffer(GL_ARRAY_BUFFER, vbo_currentarrow[2]);
    glBufferData(GL_ARRAY_BUFFER, nvalues_currentarrow.size()*4, &nvalues_currentarrow[0], GL_STATIC_DRAW);

    std::vector<glm::vec3> vert_forcearrow = forcearrow.getVertices();
    std::vector<glm::vec2> tex_forcearrow = forcearrow.getTextCoords();
    std::vector<glm::vec3> norm_forcearrow = forcearrow.getNormalVecs();
    numObjVertices = forcearrow.getNumVertices();

    std::vector<float> pvalues_forcearrow; //vertex positions
    std::vector<float> tvalues_forcearrow; //texture coordinates
    std::vector<float> nvalues_forcearrow; //normal vectors

    for (int i=0; i<numObjVertices; i++){
        pvalues_forcearrow.push_back((vert_forcearrow[i]).x);
        pvalues_forcearrow.push_back((vert_forcearrow[i]).y);
        pvalues_forcearrow.push_back((vert_forcearrow[i]).z);
        tvalues_forcearrow.push_back((tex_forcearrow[i]).s);
        tvalues_forcearrow.push_back((tex_forcearrow[i]).t);
        nvalues_forcearrow.push_back((norm_forcearrow[i]).x);
        nvalues_forcearrow.push_back((norm_forcearrow[i]).y);
        nvalues_forcearrow.push_back((norm_forcearrow[i]).z);
    }
    glGenVertexArrays(1, vao);
    glBindVertexArray(vao[0]);
    glGenBuffers(numVBOs, vbo_forcearrow);

    //VBO for vertex location
    glBindBuffer(GL_ARRAY_BUFFER, vbo_forcearrow[0]);
    glBufferData(GL_ARRAY_BUFFER, pvalues_forcearrow.size()*4, &pvalues_forcearrow[0], GL_STATIC_DRAW);

    //VBO for texture coordinates
    glBindBuffer(GL_ARRAY_BUFFER, vbo_forcearrow[1]);
    glBufferData(GL_ARRAY_BUFFER, tvalues_forcearrow.size()*4, &tvalues_forcearrow[0], GL_STATIC_DRAW);

    //VBO for normal vectors
    glBindBuffer(GL_ARRAY_BUFFER, vbo_forcearrow[2]);
    glBufferData(GL_ARRAY_BUFFER, nvalues_forcearrow.size()*4, &nvalues_forcearrow[0], GL_STATIC_DRAW);

    std::vector<glm::vec3> vert_rudder= rudder.getVertices();
    std::vector<glm::vec2> tex_rudder = rudder.getTextCoords();
    std::vector<glm::vec3> norm_rudder = rudder.getNormalVecs();
    numObjVertices = rudder.getNumVertices();

    std::vector<float> pvalues_rudder; //vertex positions
    std::vector<float> tvalues_rudder; //texture coordinates
    std::vector<float> nvalues_rudder; //normal vectors

    for (int i=0; i<numObjVertices; i++){
        pvalues_rudder.push_back((vert_rudder[i]).x);
        pvalues_rudder.push_back((vert_rudder[i]).y);
        pvalues_rudder.push_back((vert_rudder[i]).z);
        tvalues_rudder.push_back((tex_rudder[i]).s);
        tvalues_rudder.push_back((tex_rudder[i]).t);
        nvalues_rudder.push_back((norm_rudder[i]).x);
        nvalues_rudder.push_back((norm_rudder[i]).y);
        nvalues_rudder.push_back((norm_rudder[i]).z);
    }
    glGenVertexArrays(1, vao);
    glBindVertexArray(vao[0]);
    glGenBuffers(numVBOs, vbo_rudder);

    //VBO for vertex location
    glBindBuffer(GL_ARRAY_BUFFER, vbo_rudder[0]);
    glBufferData(GL_ARRAY_BUFFER, pvalues_rudder.size()*4, &pvalues_rudder[0], GL_STATIC_DRAW);

    //VBO for texture coordinates
    glBindBuffer(GL_ARRAY_BUFFER, vbo_rudder[1]);
    glBufferData(GL_ARRAY_BUFFER, tvalues_rudder.size()*4, &tvalues_rudder[0], GL_STATIC_DRAW);

    //VBO for normal vectors
    glBindBuffer(GL_ARRAY_BUFFER, vbo_rudder[2]);
    glBufferData(GL_ARRAY_BUFFER, nvalues_rudder.size()*4, &nvalues_rudder[0], GL_STATIC_DRAW);

    std::vector<glm::vec3> vert_clock_momentum = clock_momentum.getVertices();
    std::vector<glm::vec2> tex_clock_momentum = clock_momentum.getTextCoords();
    std::vector<glm::vec3> norm_clock_momentum = clock_momentum.getNormalVecs();
    numObjVertices = clock_momentum.getNumVertices();

    std::vector<float> pvalues_clock_momentum; //vertex positions
    std::vector<float> tvalues_clock_momentum; //texture coordinates
    std::vector<float> nvalues_clock_momentum; //normal vectors

    for (int i=0; i<numObjVertices; i++){
        pvalues_clock_momentum.push_back((vert_clock_momentum[i]).x);
        pvalues_clock_momentum.push_back((vert_clock_momentum[i]).y);
        pvalues_clock_momentum.push_back((vert_clock_momentum[i]).z);
        tvalues_clock_momentum.push_back((tex_clock_momentum[i]).s);
        tvalues_clock_momentum.push_back((tex_clock_momentum[i]).t);
        nvalues_clock_momentum.push_back((norm_clock_momentum[i]).x);
        nvalues_clock_momentum.push_back((norm_clock_momentum[i]).y);
        nvalues_clock_momentum.push_back((norm_clock_momentum[i]).z);
    }
    glGenVertexArrays(1, vao);
    glBindVertexArray(vao[0]);
    glGenBuffers(numVBOs, vbo_clock_momentum);

    //VBO for vertex location
    glBindBuffer(GL_ARRAY_BUFFER, vbo_clock_momentum[0]);
    glBufferData(GL_ARRAY_BUFFER, pvalues_clock_momentum.size()*4, &pvalues_clock_momentum[0], GL_STATIC_DRAW);

    //VBO for texture coordinates
    glBindBuffer(GL_ARRAY_BUFFER, vbo_clock_momentum[1]);
    glBufferData(GL_ARRAY_BUFFER, tvalues_clock_momentum.size()*4, &tvalues_clock_momentum[0], GL_STATIC_DRAW);

    //VBO for normal vectors
    glBindBuffer(GL_ARRAY_BUFFER, vbo_clock_momentum[2]);
    glBufferData(GL_ARRAY_BUFFER, nvalues_clock_momentum.size()*4, &nvalues_clock_momentum[0], GL_STATIC_DRAW);

    std::vector<glm::vec3> vert_anticlock_momentum = anticlock_momentum.getVertices();
    std::vector<glm::vec2> tex_anticlock_momentum = anticlock_momentum.getTextCoords();
    std::vector<glm::vec3> norm_anticlock_momentum = anticlock_momentum.getNormalVecs();
    numObjVertices = anticlock_momentum.getNumVertices();

    std::vector<float> pvalues_anticlock_momentum; //vertex positions
    std::vector<float> tvalues_anticlock_momentum; //texture coordinates
    std::vector<float> nvalues_anticlock_momentum; //normal vectors

    for (int i=0; i<numObjVertices; i++){
        pvalues_anticlock_momentum.push_back((vert_anticlock_momentum[i]).x);
        pvalues_anticlock_momentum.push_back((vert_anticlock_momentum[i]).y);
        pvalues_anticlock_momentum.push_back((vert_anticlock_momentum[i]).z);
        tvalues_anticlock_momentum.push_back((tex_anticlock_momentum[i]).s);
        tvalues_anticlock_momentum.push_back((tex_anticlock_momentum[i]).t);
        nvalues_anticlock_momentum.push_back((norm_anticlock_momentum[i]).x);
        nvalues_anticlock_momentum.push_back((norm_anticlock_momentum[i]).y);
        nvalues_anticlock_momentum.push_back((norm_anticlock_momentum[i]).z);
    }
    glGenVertexArrays(1, vao);
    glBindVertexArray(vao[0]);
    glGenBuffers(numVBOs, vbo_anticlock_momentum);

    //VBO for vertex location
    glBindBuffer(GL_ARRAY_BUFFER, vbo_anticlock_momentum[0]);
    glBufferData(GL_ARRAY_BUFFER, pvalues_anticlock_momentum.size()*4, &pvalues_anticlock_momentum[0], GL_STATIC_DRAW);

    //VBO for texture coordinates
    glBindBuffer(GL_ARRAY_BUFFER, vbo_anticlock_momentum[1]);
    glBufferData(GL_ARRAY_BUFFER, tvalues_anticlock_momentum.size()*4, &tvalues_anticlock_momentum[0], GL_STATIC_DRAW);

    //VBO for normal vectors
    glBindBuffer(GL_ARRAY_BUFFER, vbo_anticlock_momentum[2]);
    glBufferData(GL_ARRAY_BUFFER, nvalues_anticlock_momentum.size()*4, &nvalues_anticlock_momentum[0], GL_STATIC_DRAW);
}


void init (GLFWwindow* window){
    renderingProgram = createShaderProgram(concat(shader_folder, "vertShader.glsl"),concat(shader_folder, "fragShader.glsl"));
    cameraX = 30.0f; cameraY = 14.0f; cameraZ = 30.0f;
    boatLocX = 0.0f; boatLocY = 2.0f; boatLocZ = 0.0f;
    setupVertices();

    //Costruisco la pMat
    glfwGetFramebufferSize(window, &width, &height);
    aspect = (float)width / (float)height;
    pMat = glm::perspective(1.0472f, aspect, 0.1f, 1000.0f); //1.0472 radians = 60 degrees
    lookingDirX = -1; lookingDirY = 0; lookingDirZ = 0;
}

void display (GLFWwindow* window, double currentTime){
    double lambda = recived_values[4]/*cos((float)currentTime*3)*/;
    double sigma = recived_values[7];
    glClear(GL_DEPTH_BUFFER_BIT);
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT); //Clear the background to black
    glUseProgram(renderingProgram);
    //glEnable(GL_CULL_FACE);

    //Getto le uniform
    mvLoc = glGetUniformLocation(renderingProgram, "mv_matrix");
    projLoc = glGetUniformLocation(renderingProgram, "proj_matrix");
    nLoc = glGetUniformLocation(renderingProgram, "norm_matrix");

    //Aggiorno la posizione della camera
    //cameraZ = (28+20*cos((float)currentTime))*sin((float)currentTime*0.6);
    //cameraX = (28+20*cos((float)currentTime))*cos((float)currentTime*0.6);
    //cameraY = (7+5*cos((float)currentTime));

    //boatLocX = (float)(10*cos((float)currentTime*1));

    //Costruisco la mvMat
    vMat = glm::lookAt(glm::vec3(cameraX, cameraY, cameraZ), glm::vec3(cameraX+lookingDirX, cameraY+lookingDirY, cameraZ+lookingDirZ), glm::vec3(0.0f, 1.0f, 0.0f));
    //vMat = glm::lookAt(glm::vec3(cameraX+recived_values[0], cameraY, cameraZ+recived_values[1]), glm::vec3(cameraX+recived_values[0]+lookingDirX, cameraY+lookingDirY, cameraZ+recived_values[1]+lookingDirZ), glm::vec3(0.0f, 1.0f, 0.0f));
    mMat = glm::translate(glm::mat4(1.0f), glm::vec3(recived_values[0], boatLocY, recived_values[1]));
    glm::mat4 rotation_phi = glm::rotate(glm::mat4(1.0f), (float)(recived_values[2]), glm::vec3(1.0f,0.0f,0.0f));
    glm::mat4 rotation_xi = glm::rotate(glm::mat4(1.0f), (float)(recived_values[3]), glm::vec3(0.0f, 1.0f, 0.0f));
    mMat = mMat * rotation_xi;
    mMat = mMat * rotation_phi;


    //Set up lights
    currentLightPos = glm::vec3(initialLightLoc.x, initialLightLoc.y, initialLightLoc.z);
    installLights(vMat);

    mvMat = vMat * mMat;

    //Build matrix for trasforming vectors
    invTrMat = glm::transpose(glm::inverse(mvMat));

    //Spedisco matrici allo shader
    glUniformMatrix4fv(mvLoc, 1, GL_FALSE, glm::value_ptr(mvMat));
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(pMat));
    glUniformMatrix4fv(nLoc, 1, GL_FALSE, glm::value_ptr(invTrMat));

    //Associazione VBO
    glBindBuffer(GL_ARRAY_BUFFER, vbo_boat[0]);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_boat[2]);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);

    // glEnable(GL_CULL_FACE);
    glFrontFace(GL_CCW);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    glDrawArrays(GL_TRIANGLES, 0, boat.getNumVertices());

    mMat = glm::translate(glm::mat4(1.0f), glm::vec3(recived_values[0], boatLocY, recived_values[1]));
    glm::vec4 new_y = rotation_phi*glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
    glm::mat4 rotation_lambda = glm::rotate(glm::mat4(1.0f), (float)(lambda), glm::vec3(0.0f, 1.0f, 0.0f));
    mMat = mMat * rotation_xi;
    mMat = mMat * rotation_phi;
    mMat = mMat * rotation_lambda;
    mvMat = vMat * mMat;

    //Spedisco matrici allo shader
    glUniformMatrix4fv(mvLoc, 1, GL_FALSE, glm::value_ptr(mvMat));
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(pMat));
    glUniformMatrix4fv(nLoc, 1, GL_FALSE, glm::value_ptr(invTrMat));

    //Associazione VBO
    glBindBuffer(GL_ARRAY_BUFFER, vbo_sail[0]);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_sail[2]);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);

    // glEnable(GL_CULL_FACE);
    glFrontFace(GL_CCW);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    glDrawArrays(GL_TRIANGLES, 0, sail.getNumVertices());

    mMat = glm::translate(glm::mat4(1.0f), glm::vec3(recived_values[0], boatLocY, recived_values[1]));
    glm::mat4 rotation_sigma = glm::rotate(glm::mat4(1.0f), (float)(sigma-M_PI/2), glm::vec3(0.0f, 1.0f, 0.0f));
    mMat = mMat * rotation_xi;
    mMat = mMat * rotation_phi;
    mMat = mMat * rotation_sigma;
    mMat *= glm::scale(glm::mat4(1.0f), glm::vec3(1.f, 1.f, 10.f));
    mvMat = vMat * mMat;

    //Spedisco matrici allo shader
    glUniformMatrix4fv(mvLoc, 1, GL_FALSE, glm::value_ptr(mvMat));
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(pMat));
    glUniformMatrix4fv(nLoc, 1, GL_FALSE, glm::value_ptr(invTrMat));

    //Associazione VBO
    glBindBuffer(GL_ARRAY_BUFFER, vbo_rudder[0]);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_rudder[2]);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);

    // glEnable(GL_CULL_FACE);
    glFrontFace(GL_CCW);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    glDrawArrays(GL_TRIANGLES, 0, rudder.getNumVertices());

    mMat = glm::translate(glm::mat4(1.0f), glm::vec3(recived_values[0], boatLocY+10, recived_values[1]));
    rotation_lambda = glm::rotate(glm::mat4(1.0f), (float)(recived_values[5]+M_PI/2), glm::vec3(0.0f, 1.0f, 0.0f));
    mMat = mMat * rotation_lambda;
    mMat *= glm::scale(glm::mat4(1.0f), glm::vec3(0.1f, 0.1f, 0.1f));
    mvMat = vMat * mMat;

    //Spedisco matrici allo shader
    glUniformMatrix4fv(mvLoc, 1, GL_FALSE, glm::value_ptr(mvMat));
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(pMat));
    glUniformMatrix4fv(nLoc, 1, GL_FALSE, glm::value_ptr(invTrMat));

    //Associazione VBO
    glBindBuffer(GL_ARRAY_BUFFER, vbo_windarrow[0]);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_windarrow[2]);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);

    // glEnable(GL_CULL_FACE);
    glFrontFace(GL_CCW);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    glDrawArrays(GL_TRIANGLES, 0, windarrow.getNumVertices());

    mMat = glm::translate(glm::mat4(1.0f), glm::vec3(recived_values[0], boatLocY+4, recived_values[1]));
    rotation_lambda = glm::rotate(glm::mat4(1.0f), (float)(recived_values[6]+M_PI/2), glm::vec3(0.0f, 1.0f, 0.0f));
    mMat = mMat * rotation_lambda;
    mMat *= glm::scale(glm::mat4(1.0f), glm::vec3(0.1f, 0.1f, 0.1f));
    mvMat = vMat * mMat;

    //Spedisco matrici allo shader
    glUniformMatrix4fv(mvLoc, 1, GL_FALSE, glm::value_ptr(mvMat));
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(pMat));
    glUniformMatrix4fv(nLoc, 1, GL_FALSE, glm::value_ptr(invTrMat));

    //Associazione VBO
    glBindBuffer(GL_ARRAY_BUFFER, vbo_forcearrow[0]);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_forcearrow[2]);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);

    // glEnable(GL_CULL_FACE);
    glFrontFace(GL_CCW);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    glDrawArrays(GL_TRIANGLES, 0, forcearrow.getNumVertices());

    if (recived_values[3]<-10e-4){
        mMat = glm::translate(glm::mat4(1.0f), glm::vec3(recived_values[0], boatLocY, recived_values[1]));
        mMat = mMat * rotation_xi;
        mMat = mMat * rotation_phi;
        mvMat = vMat * mMat;

        //Spedisco matrici allo shader
        glUniformMatrix4fv(mvLoc, 1, GL_FALSE, glm::value_ptr(mvMat));
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(pMat));
        glUniformMatrix4fv(nLoc, 1, GL_FALSE, glm::value_ptr(invTrMat));

        //Associazione VBO
        glBindBuffer(GL_ARRAY_BUFFER, vbo_clock_momentum[0]);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(0);

        glBindBuffer(GL_ARRAY_BUFFER, vbo_clock_momentum[2]);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(1);

        // glEnable(GL_CULL_FACE);
        glFrontFace(GL_CCW);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);

        glDrawArrays(GL_TRIANGLES, 0, clock_momentum.getNumVertices());
    }
    if (recived_values[3]>10e-4){
        mMat = glm::translate(glm::mat4(1.0f), glm::vec3(recived_values[0], boatLocY, recived_values[1]));
        mMat = mMat * rotation_xi;
        mMat = mMat * rotation_phi;
        mvMat = vMat * mMat;

        //Spedisco matrici allo shader
        glUniformMatrix4fv(mvLoc, 1, GL_FALSE, glm::value_ptr(mvMat));
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(pMat));
        glUniformMatrix4fv(nLoc, 1, GL_FALSE, glm::value_ptr(invTrMat));

        //Associazione VBO
        glBindBuffer(GL_ARRAY_BUFFER, vbo_anticlock_momentum[0]);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(0);

        glBindBuffer(GL_ARRAY_BUFFER, vbo_anticlock_momentum[2]);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(1);

        // glEnable(GL_CULL_FACE);
        glFrontFace(GL_CCW);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);

        glDrawArrays(GL_TRIANGLES, 0, anticlock_momentum.getNumVertices());
    }

    mMat = glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, 0));
    mMat *= glm::scale(glm::mat4(1.0f), glm::vec3(10.f, 10.f, 10.f));
    mvMat = vMat * mMat;

    //Spedisco matrici allo shader
    glUniformMatrix4fv(mvLoc, 1, GL_FALSE, glm::value_ptr(mvMat));
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(pMat));
    glUniformMatrix4fv(nLoc, 1, GL_FALSE, glm::value_ptr(invTrMat));

    //Associazione VBO
    glBindBuffer(GL_ARRAY_BUFFER, vbo_insland[0]);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_insland[2]);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);

    // glEnable(GL_CULL_FACE);
    glFrontFace(GL_CCW);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    glDrawArrays(GL_TRIANGLES, 0, insland.getNumVertices());
}

void installLights(glm::mat4 vMat){
    //Convert light's position to view space and prepare shader data sending
    lightPosV = glm::vec3(vMat*glm::vec4(currentLightPos, 1.0));
    lightPos[0] = lightPosV.x;
    lightPos[1] = lightPosV.y;
    lightPos[2] = lightPosV.z;

    //get locations
    globalAmbLoc = glGetUniformLocation(renderingProgram, "globalAmbient");
    ambLoc = glGetUniformLocation(renderingProgram, "light.ambient");
    diffLoc = glGetUniformLocation(renderingProgram, "light.diffuse");
    specLoc = glGetUniformLocation(renderingProgram, "light.specular");
    posLoc = glGetUniformLocation(renderingProgram, "light.position");
    mAmbLoc = glGetUniformLocation(renderingProgram, "material.ambient");
    mDiffLoc = glGetUniformLocation(renderingProgram, "material.diffuse");
    mSpecLoc = glGetUniformLocation(renderingProgram, "material.specular");
    mShiLoc = glGetUniformLocation(renderingProgram, "material.shininess");

    //set uniforms
    glProgramUniform4fv(renderingProgram, globalAmbLoc, 1, globalAmbient);
    glProgramUniform4fv(renderingProgram, ambLoc, 1, lightAmbient);
    glProgramUniform4fv(renderingProgram, diffLoc, 1, lightDiffuse);
    glProgramUniform4fv(renderingProgram, specLoc, 1, lightSpecular);
    glProgramUniform3fv(renderingProgram, posLoc, 1, lightPos);
    glProgramUniform4fv(renderingProgram, mAmbLoc, 1, matAmb);
    glProgramUniform4fv(renderingProgram, mDiffLoc, 1, matDif);
    glProgramUniform4fv(renderingProgram, mSpecLoc, 1, matSpe);
    glProgramUniform1f(renderingProgram, mShiLoc, matShi);

}

void window_reshape_callback(GLFWwindow* window, int newWidth, int newHeight){
    aspect = (float)newWidth/(float)newHeight;
    glViewport(0, 0, newWidth, newHeight);
    pMat = glm::perspective(1.0472f, aspect, 0.1f, 1000.0f); //1.0472 radians = 60 degrees
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods){
    //cout<<"KEY:"<<key<<" - "<<glfwGetKeyScancode(key)<<endl;
    /*FrecciaSu : 265
     *FrecciaGiu : 264
     *FrecciaSx : 263
     *FrecciaDx : 262
    */
    // std::cout << "Premuto: " << key << std::endl;
    if (glfwGetKey(window, 81) == GLFW_PRESS){//Q
        sigma_to_send += PASSOVELA;
        if (sigma_to_send>M_PI/4){
            sigma_to_send = M_PI/4;
        }
    }
    if (glfwGetKey(window, 69) == GLFW_PRESS){//E
        sigma_to_send -= PASSOVELA;
        if (sigma_to_send<-M_PI/4){
            sigma_to_send = -M_PI/4;
        }
    }
    if (glfwGetKey(window, 265) == GLFW_PRESS){//freccia su
      cameraAttacedToBoat = true;
    }
    if (glfwGetKey(window, 264) == GLFW_PRESS){//freccia giu
      cameraAttacedToBoat = false;
    }
    if (glfwGetKey(window, 263) == GLFW_PRESS){//freccia sinistra
        lambda_to_send -= PASSOVELA;
        if (lambda_to_send<-M_PI/2){
            lambda_to_send = -M_PI/2;
        }
    }
    if (glfwGetKey(window, 262) == GLFW_PRESS){//freccia destra
        lambda_to_send += PASSOVELA;
        if (lambda_to_send>M_PI/2){
            lambda_to_send = M_PI/2;
        }
    }
    if (glfwGetKey(window, 87) == GLFW_PRESS){//W
        if (!cameraAttacedToBoat){
            cameraX += lookingDirX*PASSOCAMERA;
            cameraZ += lookingDirZ*PASSOCAMERA;
        }
    }
    if (glfwGetKey(window, 83) == GLFW_PRESS){//S
        if (!cameraAttacedToBoat){
            cameraX -= lookingDirX*PASSOCAMERA;
            cameraZ -= lookingDirZ*PASSOCAMERA;
        }
    }
    if (glfwGetKey(window, 256) == GLFW_PRESS){// esc
        if (mouseblocked){
          glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
          mouseblocked = false;
        }else{
          glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
          mouseblocked = true;
        }
    }
}

static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos){
  if (mouseblocked && !cameraAttacedToBoat){
    glm::vec3 newlookingDir(lookingDirX, lookingDirY, lookingDirZ);
    float newlookingDirX = lookingDirX;
    float newlookingDirZ = lookingDirZ;
    float newlookingDirY = lookingDirY;
    float thetax = PASSOVISUALE*abs(backmousex-xpos)/50;
    float thetay = PASSOVISUALE*abs(backmousey-ypos)/200;
    if (xpos>backmousex){
      newlookingDir = rotate(0, 1, 0, glm::vec3(lookingDirX, lookingDirY, lookingDirZ), thetax);
    }
    if (xpos<backmousex){
      newlookingDir = rotate(0, 1, 0, glm::vec3(lookingDirX, lookingDirY, lookingDirZ), -thetax);
    }
    if (ypos>backmousey && atan(lookingDirY/(sqrt(lookingDirX*lookingDirX+lookingDirZ*lookingDirZ)))>-0.785398f){
      newlookingDir = rotate(-(lookingDirZ)/sqrt(lookingDirX*lookingDirX+lookingDirZ*lookingDirZ), 0, (lookingDirX)/sqrt(lookingDirX*lookingDirX+lookingDirZ*lookingDirZ), glm::vec3(lookingDirX, lookingDirY, lookingDirZ), thetay);
    }
    if (ypos<backmousey && atan(lookingDirY/(sqrt(lookingDirX*lookingDirX+lookingDirZ*lookingDirZ)))<0.785398f){
      newlookingDir = rotate(-(lookingDirZ)/sqrt(lookingDirX*lookingDirX+lookingDirZ*lookingDirZ), 0, (lookingDirX)/sqrt(lookingDirX*lookingDirX+lookingDirZ*lookingDirZ), glm::vec3(lookingDirX, lookingDirY, lookingDirZ), -thetay);
    }
    lookingDirX = newlookingDir.x;
    lookingDirY = newlookingDir.y;
    lookingDirZ = newlookingDir.z;
    backmousex = xpos;
    backmousey = ypos;

    std::cout << "backpos: [" << backmousex << "; " << backmousey << "]" << std::endl;
    std::cout << "pos: [" << xpos << "; " << ypos << "]" << std::endl;
    std::cout << "theta: [" << thetax << "; " << thetay << "]" << std::endl;
    std::cout << "LokkingDir : [" << lookingDirX << ", " << lookingDirY << ", " << lookingDirZ << "]" << std::endl;

    //cout << xpos << ";" << ypos << endl;
  }
}

int main(void){
    MySocketClient MSC = MySocketClient();
    if (!glfwInit()) {exit(EXIT_FAILURE);}
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    GLFWwindow* window = glfwCreateWindow(600, 600, "Catamarano", NULL, NULL);
    glfwMakeContextCurrent(window);
    if (glewInit() != GLEW_OK){exit(EXIT_FAILURE);}
    glfwSwapInterval(1);

    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    mouseblocked=true;
    glfwSetWindowSizeCallback(window, window_reshape_callback);

    init(window);

    while (!glfwWindowShouldClose(window)) {
        recived_values = MSC.reading(lambda_to_send, sigma_to_send);
        display(window, glfwGetTime());
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}

