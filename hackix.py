#!/usr/bin/env python
# pygame + PyOpenGL version of Nehe's OpenGL lesson01
# Paul Furber 2001 - m@verick.co.za

from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *


def resize((width, height)):
    if height==0:
        height=1
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 1.0*width/height, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def init():
    glShadeModel(GL_SMOOTH)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
    glEnableClientState(GL_COLOR_ARRAY)
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnable(GL_CULL_FACE)


class Point:
    def __init__(self, x, y, z):
	self.x = x
	self.y = y
	self.z = z

class Face:
    def __init__(self, a, b, c, color):
	self.a = a
	self.b = b
	self.c = c
	self.color = color

class HackRender:
    def update_faces(self, faces):
	self.vertexbuffer=[]
	self.colorbuffer=[]
	self.vertexcount = 0
	for f in faces:
	    for p in (f.a, f.b, f.c):
		self.vertexbuffer.append((p.x, p.y, p.z))
		self.colorbuffer.append(f.color)
	    self.vertexcount += 3

    def draw(self, angle1, angle2):
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
	glLoadIdentity()
	glTranslatef(0.0, 0.0, -10.0)
	glRotatef(angle2, 1.0, 0.0, 0.0)
	glRotatef(angle1, 0.0, 1.0, 0.0)
	glVertexPointerf(self.vertexbuffer)
	glColorPointerf(self.colorbuffer)
	glDrawArrays(GL_TRIANGLES, 0, self.vertexcount)

def create_cube():
    points=[]
    for x in (1.0, -1.0):
	for y in (1.0, -1.0):
	    for z in (1.0, -1.0):
		points.append(Point(x, y, z))
    faces=[]
    r = (1.0, 0.0, 0.0)
    g = (0.0, 1.0, 0.0)
    b = (0.0, 0.0, 1.0)
    for (a, b, c, color) in [(0,2,3,r), (0,3,1,r), (4,5,7,r), (4,7,6,r), \
	     (0,1,5,g), (0,5,4,g), (2,6,7,g), (2,7,3,g), \
	     (0,4,6,b), (0,6,2,b), (1,3,7,b), (1,7,5,b)] :
	faces.append(Face(points[a], points[b], points[c], color))
    return faces

def main():

    video_flags = OPENGL|DOUBLEBUF
    
    pygame.init()
    pygame.display.set_mode((640,480), video_flags)

    resize((640,480))
    init()

    render = HackRender();
    frames = 0
    ticks = pygame.time.get_ticks()
    render.update_faces(create_cube())

    while 1:
        event = pygame.event.poll()
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            break
        
        render.draw(frames, frames / 10.0)
        pygame.display.flip()
        frames = frames+1

    print "fps:  %d" % ((frames*1000)/(pygame.time.get_ticks()-ticks))


if __name__ == '__main__': main()
