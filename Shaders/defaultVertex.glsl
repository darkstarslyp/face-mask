#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec2 vertexuv;

out vec2 UV;
uniform mat4 projection;

void main() {
    gl_Position = projection*vec4(position, 1.0f);
    UV = vertexuv;
}
