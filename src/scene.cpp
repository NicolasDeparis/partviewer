#include "scene.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <SDL/SDL_opengl.h>

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

void render_part(GLuint* vbo, int N){

 // printf("N=%d\n",N);

  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_DST_ALPHA	);

  glEnableClientState( GL_VERTEX_ARRAY );
  glEnableClientState( GL_COLOR_ARRAY );

  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glVertexPointer(3, GL_FLOAT, 0, NULL);

  glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
    glColorPointer(4, GL_FLOAT, 0, NULL);

  glDrawArrays(GL_POINTS, 0, N);

  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glDisableClientState( GL_COLOR_ARRAY );
  glDisableClientState( GL_VERTEX_ARRAY );
  glDisable(GL_BLEND);
}

void render(GLuint *vbo, int *N){
  render_cube();
  render_part(&vbo[0], N[0]);
  render_part(&vbo[2], N[1]);
  glFlush();
}

