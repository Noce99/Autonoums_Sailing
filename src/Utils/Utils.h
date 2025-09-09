#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>
#include <iostream>
#include <fstream>

using namespace std;

void printShaderLog(GLuint);
void printProgramLog(int);
bool checkOpenGLError();
string readShaderSource(string);

GLuint createShaderProgram(string vertShaderStrName, string fragShaderStrName){
    GLint vertCompiled;
    GLint fragCompiled;
    GLint linked;

    string vertShaderStr = readShaderSource(vertShaderStrName);
    string fragShaderStr = readShaderSource(fragShaderStrName);

    const char *vertShaderScr = vertShaderStr.c_str();
    const char *fragShaderScr = fragShaderStr.c_str();

    GLuint vShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fShader = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(vShader, 1, &vertShaderScr, NULL);
    glShaderSource(fShader, 1, &fragShaderScr, NULL);

    glCompileShader(vShader);
    checkOpenGLError();
    glGetShaderiv(vShader, GL_COMPILE_STATUS, &vertCompiled);
    if (vertCompiled != true){
        cout << "Vertex compilation FAILED!" << endl;
        printShaderLog(vShader);
    }

    glCompileShader(fShader);
    checkOpenGLError();
    glGetShaderiv(fShader, GL_COMPILE_STATUS, &fragCompiled);
    if (fragCompiled != true){
        cout << "Fragment compilation FAILED!" << endl;
        printShaderLog(fShader);
    }

    GLuint vfProgram = glCreateProgram();

    glAttachShader(vfProgram, vShader);
    glAttachShader(vfProgram, fShader);

    glLinkProgram(vfProgram);
    checkOpenGLError();
    glGetProgramiv(vfProgram, GL_LINK_STATUS, &linked);
    if (linked != true){
        cout << "Linking FAILED!" << endl;
        printProgramLog(vfProgram);
    }

    return vfProgram;
}

void printShaderLog(GLuint shader){
  int len = 0;
  int chWrittn = 0;
  char *log;
  glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
  if (len > 0){
    log = (char *)malloc(len);
    glGetShaderInfoLog(shader, len, &chWrittn, log);
    cout << "Shader Info Log:" << log << endl;
    free(log);
  }
}

void printProgramLog(int prog){
    int len = 0;
    int chWrittn = 0;
    char *log;
    glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
    if (len > 0){
      log = (char *)malloc(len);
      glGetProgramInfoLog(prog, len, &chWrittn, log);
      cout << "Program Info Log:" << log << endl;
      free(log);
    }
}

bool checkOpenGLError(){
  bool foundError = false;
  int glErr = glGetError();
  while (glErr != GL_NO_ERROR){
    cout << "GL error:" << glErr << endl;
    foundError = true;
    glErr = glGetError();
  }
  return foundError;
}

string readShaderSource(string filePath){
    string content;
    ifstream fileStream(filePath.c_str(), ios::in);
    string line = "";
    while (!fileStream.eof()){
      getline(fileStream, line);
      content.append(line + "\n");
    }
    fileStream.close();
    return content;
}

//GOLD Material - ambien, diffuse, specular and shininess
float* goldAmbient(){
  static float a[4] = {0.2473f, 0.1995f, 0.0745f, 1};
  return (float*)a;
}
float* goldDiffuse(){
  static float a[4] = {0.7516f, 0.6065f, 0.2265f, 1};
  return (float*)a;
}
float* goldSpecular(){
  static float a[4] = {0.6283f, 0.5559f, 0.3661f, 1};
  return (float*)a;
}
float goldShininess(){
  return 51.2f;
}

glm::vec3 rotate(float x, float y, float z, glm::vec3 vec, float T){
  return glm::mat3( x*x+(1-x*x)*cos(T),       x*y*(1-cos(T))-z*sin(T),   x*z*(1-cos(T))+y*sin(T),
                    x*y*(1-cos(T))+z*sin(T),  y*y+(1-y*y)*cos(T),       y*z*(1-cos(T))-x*sin(T),
                    x*z*(1-cos(T))-y*sin(T),  y*z*(1-cos(T))+x*sin(T),  z*z+(1-z*z)*cos(T)
                  )*vec;
}

string concat(const char * first, const char * second){
	return string(first) + string(second);
}
