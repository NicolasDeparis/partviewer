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


/*
  glVertexPointer(3, GL_FLOAT, 3*sizeof(float),part->getPos() );
  glEnableClientState( GL_VERTEX_ARRAY );
  glDrawArrays(GL_POINTS, 0, part->getN());
*/


}
void render_test(GLuint triangleVBO){
  //Initialise VBO - do only once, at start of program
  //Create a variable to hold the VBO identifier
  //GLuint triangleVBO;

  //Vertices of a triangle (counter-clockwise winding)
  float data[] = {1.0, 0.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 1.0};
  //try float data[] = {0.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, -1.0, 0.0}; if the above doesn't work.

  //Create a new VBO and use the variable id to store the VBO id
//  glGenBuffers(1, &triangleVBO);


  //Make the new VBO active
  glBindBuffer(GL_ARRAY_BUFFER, triangleVBO);

  //Upload vertex data to the video device
  glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW);

  //Make the new VBO active. Repeat here incase changed since initialisation
  glBindBuffer(GL_ARRAY_BUFFER, triangleVBO);

  //Draw Triangle from VBO - do each time window, view point or data changes
  //Establish its 3 coordinates per vertex with zero stride in this array; necessary here
  glVertexPointer(3, GL_FLOAT, 0, NULL);

  //Establish array contains vertices (not normals, colours, texture coords etc)
  glEnableClientState(GL_VERTEX_ARRAY);

  //Actually draw the triangle, giving the number of vertices provided
  glDrawArrays(GL_TRIANGLES, 0, sizeof(data) / sizeof(float) / 3);

  //Force display to be drawn now
  glFlush();
}
void render(Part * part, GLuint vbo){

//  render_cube();
  render_part(part, vbo);
//  render_test(vbo);
  glFlush();
}

