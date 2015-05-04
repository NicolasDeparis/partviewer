
#include "scene.h"
#include <stdio.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <math.h>

void render_cube(float s){
  glBegin(GL_LINES);
    glVertex3i(0,0,0);glVertex3i(0,0,s);
    glVertex3i(0,0,0);glVertex3i(0,s,0);
    glVertex3i(0,s,0);glVertex3i(0,s,s);
    glVertex3i(0,0,s);glVertex3i(0,s,s);

    glVertex3i(s,0,0);glVertex3i(s,0,s);
    glVertex3i(s,0,0);glVertex3i(s,s,0);
    glVertex3i(s,s,0);glVertex3i(s,s,s);
    glVertex3i(s,0,s);glVertex3i(s,s,s);

    glVertex3i(0,0,0);glVertex3i(s,0,0);
    glVertex3i(0,0,s);glVertex3i(s,0,s);
    glVertex3i(0,s,0);glVertex3i(s,s,0);
    glVertex3i(0,s,s);glVertex3i(s,s,s);
  glEnd();
}

void render_part(Part * part, float scale){
  float agemax=part->getAgeMax();

  for(int i=0; i<part->getN(); i++){
    float x = part->getX(i)*scale;
    float y = part->getY(i)*scale;
    float z = part->getZ(i)*scale;

#ifdef STARS
    float age = part->getAge(i)/ agemax;
    float r=255;
    float v=255*age;
    float b=255*age;
#endif // STARS

    glBegin(GL_POINTS);
      glColor4f(1,1,1, 0.5);
      glVertex3d(x,y,z);
       // glColor3ub(r,v,b);
    glEnd();
  }
}

void render(Part * part, float scale){

  render_cube(scale);
  render_part(part, scale);

  glFlush();
}

