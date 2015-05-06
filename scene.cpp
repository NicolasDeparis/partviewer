#include "scene.h"
#include <stdio.h>
#include <stdlib.h>

#include <GL/gl.h>
#include <GL/glu.h>
#include <math.h>



void render_cube(){
  glBegin(GL_LINES);
    glVertex3i(0,0,0);glVertex3i(0,0,1);
    glVertex3i(0,0,0);glVertex3i(0,1,0);
    glVertex3i(0,1,0);glVertex3i(0,1,1);
    glVertex3i(0,0,1);glVertex3i(0,1,1);

    glVertex3i(1,0,0);glVertex3i(1,0,1);
    glVertex3i(1,0,0);glVertex3i(1,1,0);
    glVertex3i(1,1,0);glVertex3i(1,1,1);
    glVertex3i(1,0,1);glVertex3i(1,1,1);

    glVertex3i(0,0,0);glVertex3i(1,0,0);
    glVertex3i(0,0,1);glVertex3i(1,0,1);
    glVertex3i(0,1,0);glVertex3i(1,1,0);
    glVertex3i(0,1,1);glVertex3i(1,1,1);
  glEnd();
}

void render_part(Part *part, GLuint vbo){


glBindBuffer(GL_ARRAY_BUFFER, vbo);
glEnableClientState( GL_VERTEX_ARRAY );

    glVertexPointer(3, GL_FLOAT, 0, NULL);
    glDrawArrays(GL_POINTS, 0, part->getN() );

glDisableClientState( GL_VERTEX_ARRAY );
glBindBuffer(GL_ARRAY_BUFFER, 0);


}

void render(Part * part, GLuint vbo){

  render_cube();
  render_part(part, vbo);

  glFlush();
}

