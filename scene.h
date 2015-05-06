#ifndef SCENE_H
#define SCENE_H
#ifdef __APPLE__
#include <OpenGL/gl.h>
#elif
#include <GL/gl.h>
#endif
#include "Part.h"

void render(Part * parts, GLuint vbo);

#endif //SCENE_H
