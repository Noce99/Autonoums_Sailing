#version 430
layout (location=0) in vec3 vertPos;
uniform mat4 mv_matrix;
uniform mat4 proj_matrix;

out vec4 varyingColor;

void main(void){
  gl_Position = proj_matrix * mv_matrix * vec4(vertPos, 1.0);
  varyingColor = vec4(0.0, 1.0, 0.0, 1.0);
}
