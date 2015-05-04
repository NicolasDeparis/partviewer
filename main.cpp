#include <SDL/SDL.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <cstdlib>

//#include "sdlglutils.h"
#include "freeflycamera.h"
#include "scene.h"
#include "Part.h"

#define FPS 50
#define LARGEUR_FENETRE 800
#define HAUTEUR_FENETRE 600



char folder[128] = "/home/deparis/data/FF_cond150/data/";
const int  nproc = 64;
const int anim_time = 90;
const int scale = 128;

//const int nstep = 1; int num[nstep]={10};
//const int nstep = 2; int num[nstep]={0,10};
const int nstep = 6; int num[nstep]={0,2,4,6,8,10};
//const int nstep = 11; int num[nstep]={0,1,2,3,4,5,6,7,8,9,10};
//const int nstep = 6; int num[nstep]={5,6,7,8,9,10};


void DrawGL(Part * part);

FreeFlyCamera * camera;

void stop()
{
    delete camera;
    SDL_Quit();
}

int main(int argc, char *argv[]){

    int  fileNumber;
    int n=128;
    int time_max = anim_time * 1000/ nstep;
////////////////////////////////////////////////////////////////////////////////////////
    printf("Initializing\n");
////////////////////////////////////////////////////////////////////////////////////////

    Part **all_part =  (Part**)calloc(nstep,sizeof(Part*));
    float *dt =  (float*)calloc(nstep,sizeof(float));
    float *t_yr =  (float*)calloc(nstep,sizeof(float));
    float *Npart=  (float*)calloc(nstep,sizeof(float));

    for(int i=0; i<nstep; i++){
      fileNumber = num[i];
      Part* part_current = new Part(folder, fileNumber, nproc);
//      for (int ii=0;ii<100; ii++) printf("id %d\n",part_current->getIdx(ii));
 //     abort();
      Part  part_tmp(n*n*n);
      part_tmp.sort(part_current);
      part_current->copy(&part_tmp);

      all_part[i]=part_current;

      t_yr[i]= a2t(part_current->getA());
      Npart[i] = part_current->getN();


    }

    for(int i=0; i<nstep-1; i++){
        dt[i] = getdt(all_part[i], all_part[i+1])/time_max;

        all_part[i]->setV(all_part[i+1], time_max);
    //    all_part[i+1]->interpPos(all_part[i], t_yr, i, time_max);
    }

////////////////////////////////////////////////////////////////////////////////////////
    printf("Ok let's GO\n");
////////////////////////////////////////////////////////////////////////////////////////

    SDL_Event event;
    const Uint32 time_per_frame = 1000/FPS;
    unsigned int width = LARGEUR_FENETRE;
    unsigned int height = HAUTEUR_FENETRE;

    Uint32 last_time,current_time,elapsed_time, total_time; //for time animation
    Uint32 start_time,stop_time; //for frame limit

    SDL_Init(SDL_INIT_VIDEO);
    atexit(stop);

    SDL_WM_SetCaption("Particle Viewer", NULL);
    SDL_SetVideoMode(width, height, 32, SDL_OPENGL);
//    initFullScreen(&width,&height);

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    gluPerspective(70,(double)width/height,0.001,1000);

    //glEnable(GL_DEPTH_TEST);
    //glEnable(GL_BLEND);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    //glEnable(GL_POINT_SMOOTH);


    camera = new FreeFlyCamera(Vector3D(0.5*scale,0.5*scale,0.5*scale));

    last_time = SDL_GetTicks();

    start_time = SDL_GetTicks();

    int current_step=0;
    for (;;)
    {
        current_time = SDL_GetTicks();
        elapsed_time = current_time - last_time;
        last_time= current_time;
        total_time=current_time-start_time;

////////////////////////////////////////////////////////////////////////////////////////
        Part *part_current, *part_next;

        if( total_time >= time_max*current_step && current_step<nstep){
          printf("time=%d step=%d \n", total_time, current_step);
          part_current = all_part[current_step];
          part_next = all_part[current_step+1];
          current_step++;
        }

          float cur_t = t_yr[current_step]+total_time*dt[current_step];
       //   Npart[current_step]+=part_current->append(part_next, Npart[current_step], cur_t);
      //  printf("t=%f npart=%d currentN=%d\n",t_yr, Npart, part_current->getN());

////////////////////////////////////////////////////////////////////////////////////////

      //  start_time = SDL_GetTicks();

        while(SDL_PollEvent(&event))
        {
            switch(event.type)
            {
                case SDL_QUIT:
                exit(0);
                break;
                case SDL_KEYDOWN:
                switch (event.key.keysym.sym)
                {
                    case SDLK_p:
                    break;
                    case SDLK_ESCAPE:
                    exit(0);
                    break;
                    default :
                    camera->OnKeyboard(event.key);
                }
                break;
                case SDL_KEYUP:
                camera->OnKeyboard(event.key);
                break;
                case SDL_MOUSEMOTION:
                camera->OnMouseMotion(event.motion);
                break;
                case SDL_MOUSEBUTTONUP:
                case SDL_MOUSEBUTTONDOWN:
                camera->OnMouseButton(event.button);
                break;
            }
        }



        camera->animate(elapsed_time);
        if (current_time< (nstep-1)*time_max) part_current->move(elapsed_time);

        DrawGL(part_current);

        stop_time = SDL_GetTicks();
        if ((stop_time - last_time) < time_per_frame){
            SDL_Delay(time_per_frame - (stop_time - last_time));
        }
    }

    return 0;
}

void DrawGL(Part *parts)
{
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity( );

    camera->look();

    render(parts, scale);

    glFlush();
    SDL_GL_SwapBuffers();
}


