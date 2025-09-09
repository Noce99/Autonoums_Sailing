#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "Utils.h"
#include "sphere.cpp"
#include "texture_loading.cpp"
#include <opencv2/opencv.hpp>

using namespace std;

const char  height_images_folder []  = "./resources/height_images/";
const char  shader_folder []  = "./shaders/";
const char  photos_folder []  = "./resources/satellite_photos/";

#define numVAOs 2
#define numVBOs 4
//VAO = Vertex Array Objects
//VBO = Vertex Buffer Objects

#define PASSOVISUALE 0.1f
#define PASSOCAMERA 0.8f
#define PASSOZOOM 0.1f

#define LIGHTDISTANCE 5.

float cameraX , cameraY, cameraZ;
float lookingDirX, lookingDirY, lookingDirZ;
float lambda, delta;
float zoom = 3;
float tmp_X, tmp_Z;

float backmousex=100000, backmousey=0;
bool mouseblocked = false;

glm::vec3 upCamera;
GLuint renderingProgram; //GLuint è una shortcat per unsigned int
GLuint simple_renderingProgram;
GLuint textureProgram;
GLuint vao[numVAOs];
vector<GLuint [numVBOs]> vbos(20);
vector<int> vbos_num_vertices(20);
int filled_vbo = 0;

//Variabili allocate in init così non devono essere allocate durante il rendering
GLuint mvLoc, projLoc, nLoc, simple_mvLoc, simple_projLoc, texture_mvLoc, texture_projLoc, texture_nLoc;

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
glm::vec3 initialLightLoc = glm::vec3(2.f, 0.0f, 0.0f);// Initial Light position

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

Sphere sphere = Sphere(1, 100);
Cube cube = Cube(1, 1);

std::vector<GLuint> textures_id;
void textures();

void shape_setup_vertices(Cube c){
    std::vector<glm::vec3> vert = c.getVertices();
    std::vector<glm::vec3> norm = c.getNormalVecs();
    std::vector<glm::vec2> text = c.getTexture();
    int numVertices = c.getNumVertices();

    std::vector<float> pvalues; //vertex positions
    std::vector<float> tvalues; //texture coordinates
    std::vector<float> nvalues; //normal vectors
	
    for (int i=0; i<numVertices; i++){
        pvalues.push_back((vert[i]).x);
        pvalues.push_back((vert[i]).y);
        pvalues.push_back((vert[i]).z);
        tvalues.push_back((text[i]).x);
        tvalues.push_back((text[i]).y);
        nvalues.push_back((norm[i]).x);
        nvalues.push_back((norm[i]).y);
        nvalues.push_back((norm[i]).z);
    }
    glGenVertexArrays(1, vao);
    glBindVertexArray(vao[0]);
    glGenBuffers(numVBOs, vbos[filled_vbo]);

    //VBO for vertex location
    glBindBuffer(GL_ARRAY_BUFFER, vbos[filled_vbo][0]);
    glBufferData(GL_ARRAY_BUFFER, pvalues.size()*4, &pvalues[0], GL_STATIC_DRAW);

    //VBO for normal vectors
    glBindBuffer(GL_ARRAY_BUFFER, vbos[filled_vbo][1]);
    glBufferData(GL_ARRAY_BUFFER, nvalues.size()*4, &nvalues[0], GL_STATIC_DRAW);
    
    //VBO for texture vectors
    glBindBuffer(GL_ARRAY_BUFFER, vbos[filled_vbo][2]);
    glBufferData(GL_ARRAY_BUFFER, tvalues.size()*4, &tvalues[0], GL_STATIC_DRAW);
    
    vbos_num_vertices[filled_vbo] = numVertices;
    filled_vbo++;
}

void setupVertices(void){
    shape_setup_vertices(sphere);
    shape_setup_vertices(cube);
}


void init (GLFWwindow* window){
	textures();
    renderingProgram = createShaderProgram(concat(shader_folder, "vertShader.glsl"),concat(shader_folder, "fragShader.glsl"));
    simple_renderingProgram = createShaderProgram(concat(shader_folder, "simple_vertShader.glsl"),concat(shader_folder, "simple_fragShader.glsl"));
    textureProgram = createShaderProgram(concat(shader_folder, "texture_vertShader.glsl"),concat(shader_folder, "texture_fragShader.glsl"));
    cameraX = 0.0f; cameraY = 0.0f; cameraZ = -8.0f;
    setupVertices();

    //Costruisco la pMat
    glfwGetFramebufferSize(window, &width, &height);
    aspect = (float)width / (float)height;
    pMat = glm::perspective(1.0472f, aspect, 0.1f, 1000.0f); //1.0472 radians = 60 degrees
    lookingDirX = 0; lookingDirY = 0; lookingDirZ = 1;
    lambda = 0; delta = 0;
}
    
void display_vbo(int i, double x, double y, double z){
    glUseProgram(renderingProgram);
    mMat = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z));
    mvMat = vMat * mMat;

    //Build matrix for trasforming vectors
    invTrMat = glm::transpose(glm::inverse(mvMat));

    //Spedisco matrici allo shader
    glUniformMatrix4fv(mvLoc, 1, GL_FALSE, glm::value_ptr(mvMat));
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(pMat));
    glUniformMatrix4fv(nLoc, 1, GL_FALSE, glm::value_ptr(invTrMat));

    //Associazione VBO
    glBindBuffer(GL_ARRAY_BUFFER, vbos[i][0]);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, vbos[i][1]);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);

    // glEnable(GL_CULL_FACE);
    glFrontFace(GL_CCW);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    glDrawArrays(GL_TRIANGLES, 0, vbos_num_vertices[i]);
}

void simple_display_vbo(int i, double x, double y, double z, double scale){
    glUseProgram(simple_renderingProgram);
    mMat = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z));
    mMat *= glm::scale(glm::mat4(1.0f), glm::vec3(scale, scale, scale));
    mvMat = vMat * mMat;

    //Build matrix for trasforming vectors
    invTrMat = glm::transpose(glm::inverse(mvMat));

    //Spedisco matrici allo shader
    glUniformMatrix4fv(simple_mvLoc, 1, GL_FALSE, glm::value_ptr(mvMat));
    glUniformMatrix4fv(simple_projLoc, 1, GL_FALSE, glm::value_ptr(pMat));

    //Associazione VBO
    glBindBuffer(GL_ARRAY_BUFFER, vbos[i][0]);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, vbos[i][1]);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);

    // glEnable(GL_CULL_FACE);
    glFrontFace(GL_CCW);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    glDrawArrays(GL_TRIANGLES, 0, vbos_num_vertices[i]);
}

void display_vbo_texture(int i, GLuint tex_id, double x, double y, double z){
    glUseProgram(textureProgram);
    mMat = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z));
    mvMat = vMat * mMat;

    //Build matrix for trasforming vectors
    invTrMat = glm::transpose(glm::inverse(mvMat));

    //Spedisco matrici allo shader
    glUniformMatrix4fv(texture_mvLoc, 1, GL_FALSE, glm::value_ptr(mvMat));
    glUniformMatrix4fv(texture_projLoc, 1, GL_FALSE, glm::value_ptr(pMat));
    glUniformMatrix4fv(texture_nLoc, 1, GL_FALSE, glm::value_ptr(invTrMat));

    //Associazione VBO
    glBindBuffer(GL_ARRAY_BUFFER, vbos[i][0]);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, vbos[i][1]);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, vbos[i][2]);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(2);
	
	glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_id);
    
    // glEnable(GL_CULL_FACE);
    glFrontFace(GL_CCW);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    glDrawArrays(GL_TRIANGLES, 0, vbos_num_vertices[i]);
}

void display (GLFWwindow* window, double currentTime){
	glClear(GL_DEPTH_BUFFER_BIT);
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT); //Clear the background to black
	glUseProgram(renderingProgram);
	//glEnable(GL_CULL_FACE);

	//Getto le uniform
	mvLoc = glGetUniformLocation(renderingProgram, "mv_matrix");
	projLoc = glGetUniformLocation(renderingProgram, "proj_matrix");
	nLoc = glGetUniformLocation(renderingProgram, "norm_matrix");
	simple_mvLoc = glGetUniformLocation(simple_renderingProgram, "mv_matrix");
	simple_projLoc = glGetUniformLocation(simple_renderingProgram, "proj_matrix");
	texture_mvLoc = glGetUniformLocation(textureProgram, "mv_matrix");
	texture_projLoc = glGetUniformLocation(textureProgram, "proj_matrix");
	texture_nLoc = glGetUniformLocation(textureProgram, "norm_matrix");

	//Aggiorno la posizione della camera
	//cameraZ = (28+20*cos((float)currentTime))*sin((float)currentTime*0.6);
	//cameraX = (28+20*cos((float)currentTime))*cos((float)currentTime*0.6);
	//cameraY = (7+5*cos((float)currentTime));

	//boatLocX = (float)(10*cos((float)currentTime*1));

	//Costruisco la mvMat
	tmp_X = cos(lambda);
	tmp_Z = sin(lambda);
	cameraX = zoom*tmp_X*(cos(delta)+tmp_Z*tmp_Z*(1-cos(delta)))-zoom*tmp_Z*tmp_Z*tmp_X*(1-cos(delta));
	cameraY = zoom*tmp_X*tmp_X*sin(delta)+zoom*tmp_Z*tmp_Z*sin(delta);
	cameraZ = -zoom*tmp_X*tmp_X*tmp_Z*(1-cos(delta))+zoom*tmp_Z*(cos(delta)+tmp_X*tmp_X*(1-cos(delta)));
	   
		
	vMat = glm::lookAt(glm::vec3(cameraX, cameraY, cameraZ), glm::vec3(0, 0, 0), glm::vec3(0.0f, 1.0f, 0.0f));

	// vMat = glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, -8));
	//glm::mat4 rotation_phi = glm::rotate(glm::mat4(1.0f), (float)(recived_values[2]), glm::vec3(1.0f,0.0f,0.0f));
	//glm::mat4 rotation_xi = glm::rotate(glm::mat4(1.0f), (float)(recived_values[3]), glm::vec3(0.0f, 1.0f, 0.0f));
	//mMat = mMat * rotation_xi;
	//mMat = mMat * rotation_phi;


	//Set up lights
	//currentLightPos = glm::vec3(cos((float)currentTime*5.)*initialLightLoc.x-sin((float)currentTime*5.)*initialLightLoc.z, initialLightLoc.y,
	//                            cos((float)currentTime*5.)*initialLightLoc.z+sin((float)currentTime*5.)*initialLightLoc.x);
	currentLightPos = glm::vec3(LIGHTDISTANCE*tmp_X*(cos(delta)+tmp_Z*tmp_Z*(1-cos(delta)))-LIGHTDISTANCE*tmp_Z*tmp_Z*tmp_X*(1-cos(delta)),
		 							LIGHTDISTANCE*tmp_X*tmp_X*sin(delta)+LIGHTDISTANCE*tmp_Z*tmp_Z*sin(delta),
		 							-LIGHTDISTANCE*tmp_X*tmp_X*tmp_Z*(1-cos(delta))+LIGHTDISTANCE*tmp_Z*(cos(delta)+tmp_X*tmp_X*(1-cos(delta))));
	installLights(vMat);

	//display_vbo(0, 0, 0, 0);
	display_vbo_texture(0, textures_id[0], 0, 0, 0);
	//display_vbo_texture(1, 0, 0, 0);
	simple_display_vbo(1, sphere.get_index(0)[0], sphere.get_index(0)[1], sphere.get_index(0)[2], 0.01);
	simple_display_vbo(1, sphere.get_index(1000)[0], sphere.get_index(1000)[1], sphere.get_index(1000)[2], 0.01);
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

    }
    if (glfwGetKey(window, 69) == GLFW_PRESS){//E

    }
    if (glfwGetKey(window, 265) == GLFW_PRESS){//freccia su
		delta -= PASSOVISUALE;
		if (delta < -M_PI/2){
			delta = -M_PI/2+0.001;
		}
    }
    if (glfwGetKey(window, 264) == GLFW_PRESS){//freccia giu
		delta += PASSOVISUALE;
		if (delta > M_PI/2){
			delta = M_PI/2-0.001;
		}
    }
    if (glfwGetKey(window, 263) == GLFW_PRESS){//freccia sinistra
		lambda += PASSOVISUALE;
    }
    if (glfwGetKey(window, 262) == GLFW_PRESS){//freccia destra
		lambda -= PASSOVISUALE;
    }
    if (glfwGetKey(window, 87) == GLFW_PRESS){//W
        cameraX += lookingDirX*PASSOCAMERA;
        cameraZ += lookingDirZ*PASSOCAMERA;
    }
    if (glfwGetKey(window, 83) == GLFW_PRESS){//S
        cameraX -= lookingDirX*PASSOCAMERA;
        cameraZ -= lookingDirZ*PASSOCAMERA;
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

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods){
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE){
    	backmousex = 100000;
    }
}

float thetax, thetay;
static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos){
	if (mouseblocked && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==1){
		if (backmousex != 100000){
			thetax = PASSOVISUALE*(backmousex-xpos)/50;
			thetay = PASSOVISUALE*(backmousey-ypos)/200;
			delta -= thetay;
			lambda -= thetax;
			if (delta < -M_PI/2){
				delta = -M_PI/2+0.001;
			}else if (delta > M_PI/2){
				delta = M_PI/2-0.001;
			}
		}
		backmousex = xpos;
		backmousey = ypos;
	}
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset){
	zoom += yoffset*PASSOZOOM;
	if (zoom < 1.1){
		zoom = 1.1;
	}
}

void textures(){
	preLoadTexture(concat(photos_folder, "Planisfero_Carino.png"));
	preLoadTexture(concat(photos_folder, "Planisfero.jpg"));
	textures_id = loadTextures();
}

int main(void){
	//my_fork_pipe_test();
	//exit(1);

	/*cv::Mat image;
    image = cv::imread(concat(height_images_folder, "1.png"), 1 );
    if ( !image.data ){
        printf("No image data \n");
        return -1;
    }
    cv::Mat small_image = cv::Mat(image, cv::Range(3000, 4000), cv::Range(5000, 7000));
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);*/
    //cv::imshow("Display Image", small_image);
    //cv::waitKey(0);

	
    if (!glfwInit()) {exit(EXIT_FAILURE);}
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    GLFWwindow* window = glfwCreateWindow(600, 600, "Sphere", NULL, NULL);
    glfwMakeContextCurrent(window);
    if (glewInit() != GLEW_OK){exit(EXIT_FAILURE);}
    glfwSwapInterval(1);

    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    mouseblocked=true;
    glfwSetWindowSizeCallback(window, window_reshape_callback);

    init(window);
    while (!glfwWindowShouldClose(window)) {
        display(window, glfwGetTime());
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}
